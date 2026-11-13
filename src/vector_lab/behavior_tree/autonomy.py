"""Vector embodied autonomy Behavior Tree: safety, dock charging, intents, social & patrol."""

from __future__ import annotations

import math

import numpy as np

from vector_lab.behavior_tree.nodes import Action, BTNode, Condition, NodeStatus, Selector, Sequence
from vector_lab.hal.base import VectorRobotHAL
from vector_lab.sim.planner import AStarPlanner
from vector_lab.types import Blackboard, IntentType, Pose2D, Twist2D


def compute_unicycle_command(
    current: Pose2D,
    target: Pose2D,
    max_linear_mps: float = 0.16,
    max_angular_rps: float = 2.4,
) -> Twist2D:
    """Compute smooth proportional turn-and-drive unicycle velocity toward target Pose2D."""
    dist = current.distance_to(target)
    if dist < 0.025:
        return Twist2D(v=0.0, omega=0.0)

    heading_err = current.heading_error_to(target)
    omega = float(np.clip(2.8 * heading_err, -max_angular_rps, max_angular_rps))
    if abs(heading_err) > 0.55:
        return Twist2D(v=0.0, omega=omega)

    align_scale = max(0.0, math.cos(heading_err))
    dist_scale = min(1.0, dist / 0.16)
    v = float(np.clip(max_linear_mps * align_scale * max(0.35, dist_scale), 0.0, max_linear_mps))
    return Twist2D(v=v, omega=omega)


class VectorAutonomyController:
    """Stateful action callbacks bound to a VectorRobotHAL and AStarPlanner."""

    def __init__(self, hal: VectorRobotHAL, planner: AStarPlanner | None = None) -> None:
        self.hal = hal
        self.planner = planner or AStarPlanner(inflation_radius_m=0.065)

    def _plan_and_drive(
        self,
        bb: Blackboard,
        goal: Pose2D,
        branch_name: str,
        arrival_tol_m: float = 0.08,
    ) -> NodeStatus:
        bb.active_branch = branch_name
        pose = bb.estimated_pose

        if pose.distance_to(goal) <= arrival_tol_m:
            self.hal.stop_motors()
            bb.planned_path = []
            bb.active_goal = None
            return NodeStatus.SUCCESS

        needs_replan = (
            bb.active_goal is None
            or bb.active_goal.distance_to(goal) > 0.04
            or not bb.planned_path
            or bb.path_cursor >= len(bb.planned_path)
            or (bb.tick_count % 25 == 0)
        )
        if needs_replan:
            if bb.slam_map is not None:
                path = self.planner.plan(bb.slam_map, pose, goal)
                bb.planned_path = path if path else [goal]
            else:
                bb.planned_path = [goal]
            bb.active_goal = goal
            bb.path_cursor = 0

        while bb.path_cursor < len(bb.planned_path) - 1:
            wp = bb.planned_path[bb.path_cursor]
            if pose.distance_to(wp) <= 0.065:
                bb.path_cursor += 1
            else:
                break

        target_wp = bb.planned_path[min(bb.path_cursor, len(bb.planned_path) - 1)]
        twist = compute_unicycle_command(pose, target_wp)
        self.hal.drive_twist(twist.v, twist.omega)
        return NodeStatus.RUNNING

    # --- 1. Cliff Safety ---
    @staticmethod
    def is_cliff_active(bb: Blackboard) -> bool:
        if bb.cliff_recovery_ticks > 0:
            return True
        return bb.telemetry is not None and bb.telemetry.cliff.any_triggered

    def action_cliff_recover(self, bb: Blackboard) -> NodeStatus:
        bb.active_branch = "CLIFF_AVOID"
        if bb.cliff_recovery_ticks == 0:
            bb.cliff_recovery_ticks = 6
            bb.log_event("SAFETY:Cliff detected — executing reverse recovery")
            self.hal.set_eye_color(0.0, 1.0)
            self.hal.say_text("Cliff detected, backing up.")
        bb.planned_path = []
        bb.active_goal = None
        bb.cliff_recovery_ticks -= 1
        if bb.cliff_recovery_ticks > 2:
            self.hal.drive_twist(-0.10, 0.0)
        elif bb.cliff_recovery_ticks > 0:
            self.hal.drive_twist(0.0, 2.2)
        else:
            self.hal.stop_motors()
            if bb.active_intent and bb.active_intent.intent == IntentType.GOTO_POSE:
                bb.active_intent = None
                bb.log_event("SAFETY:Aborted GOTO_POSE due to cliff boundary")
            elif bb.patrol_waypoints:
                bb.patrol_index = (bb.patrol_index + 1) % len(bb.patrol_waypoints)
                bb.log_event("SAFETY:Skipped hazardous patrol waypoint after cliff recovery")
            return NodeStatus.SUCCESS
        return NodeStatus.RUNNING

    # --- 2. Operator Stop ---
    @staticmethod
    def is_operator_halted(bb: Blackboard) -> bool:
        if bb.pending_intent and bb.pending_intent.intent == IntentType.STOP:
            return True
        return bb.halted_by_operator

    def action_hold_stop(self, bb: Blackboard) -> NodeStatus:
        bb.active_branch = "OPERATOR_STOP"
        if bb.pending_intent and bb.pending_intent.intent == IntentType.STOP:
            bb.pending_intent = None
            bb.active_intent = None
            bb.halted_by_operator = True
            bb.planned_path = []
            bb.active_goal = None
            self.hal.set_eye_color(0.12, 1.0)
            self.hal.say_text("Stopping now.")
            bb.log_event("INTENT_EXEC:Operator stop engaged")
        self.hal.stop_motors()
        return NodeStatus.RUNNING

    # --- 3. Forward Obstacle Safety ---
    @staticmethod
    def is_obstacle_too_close(bb: Blackboard) -> bool:
        if bb.telemetry is not None and bb.telemetry.on_charger:
            return False
        if bb.active_branch == "DOCK_CHARGE" and bb.estimated_pose.distance_to(bb.dock_pose) < 0.24:
            return False
        return bb.forward_obstacle_dist_m < bb.safe_obstacle_dist_m

    def action_avoid_obstacle(self, bb: Blackboard) -> NodeStatus:
        bb.active_branch = "OBSTACLE_AVOID"
        self.hal.set_eye_color(0.08, 1.0)
        turn_dir = 1.0
        if bb.scan is not None:
            left_mask = (bb.scan.angles_rad > 0.1) & (bb.scan.angles_rad < 1.3)
            right_mask = (bb.scan.angles_rad < -0.1) & (bb.scan.angles_rad > -1.3)
            left_clearance = (
                float(np.mean(bb.scan.ranges_m[left_mask])) if np.any(left_mask) else 1.0
            )
            right_clearance = (
                float(np.mean(bb.scan.ranges_m[right_mask])) if np.any(right_mask) else 1.0
            )
            turn_dir = 1.0 if left_clearance >= right_clearance else -1.0

        bb.planned_path = []
        bb.active_goal = None
        self.hal.drive_twist(-0.04, turn_dir * 2.4)
        if bb.tick_count % 10 == 1:
            bb.log_event(
                f"SAFETY:Obstacle at {bb.forward_obstacle_dist_m:.2f}m — steering clear"
            )
        return NodeStatus.RUNNING

    # --- 4. Energy Management & Dock Charging ---
    @staticmethod
    def needs_dock_charge(bb: Blackboard) -> bool:
        if bb.pending_intent and bb.pending_intent.intent == IntentType.RETURN_TO_DOCK:
            return True
        if bb.active_intent and bb.active_intent.intent == IntentType.RETURN_TO_DOCK:
            return True
        if bb.telemetry is None:
            return False
        if bb.telemetry.battery_pct <= bb.low_battery_threshold_pct:
            return True
        return (
            bb.active_branch == "DOCK_CHARGE"
            and bb.telemetry.battery_pct < bb.charged_threshold_pct
        )

    def action_dock_and_charge(self, bb: Blackboard) -> NodeStatus:
        bb.active_branch = "DOCK_CHARGE"
        if bb.pending_intent and bb.pending_intent.intent == IntentType.RETURN_TO_DOCK:
            bb.active_intent = bb.pending_intent
            bb.pending_intent = None
            bb.log_event("ENERGY:Dock return requested by intent")

        tel = bb.telemetry
        if tel is not None and tel.on_charger:
            self.hal.stop_motors()
            self.hal.set_head_angle(0.0)
            self.hal.set_lift_height(0.0)
            self.hal.set_eye_color(0.35, 1.0)
            if tel.battery_pct >= bb.charged_threshold_pct:
                bb.active_intent = None
                bb.active_branch = "PATROL"
                bb.log_event(f"ENERGY:Charged to {tel.battery_pct:.1f}% — resuming patrol")
                return NodeStatus.SUCCESS
            if bb.tick_count % 25 == 0:
                bb.log_event(f"ENERGY:Charging on dock ({tel.battery_pct:.1f}%)")
            return NodeStatus.RUNNING

        dist_dock = bb.estimated_pose.distance_to(bb.dock_pose)
        dist_approach = bb.estimated_pose.distance_to(bb.dock_approach_pose)
        if dist_dock <= 0.24 or dist_approach <= 0.12:
            twist = compute_unicycle_command(bb.estimated_pose, bb.dock_pose, max_linear_mps=0.12)
            self.hal.drive_twist(twist.v, twist.omega)
            return NodeStatus.RUNNING

        self.hal.set_eye_color(0.16, 1.0)
        return self._plan_and_drive(bb, bb.dock_approach_pose, "DOCK_CHARGE", arrival_tol_m=0.09)

    # --- 5. Voice Intent Dispatch ---
    @staticmethod
    def has_voice_intent(bb: Blackboard) -> bool:
        if bb.pending_intent is not None:
            return bb.pending_intent.intent not in {IntentType.STOP, IntentType.RETURN_TO_DOCK}
        if bb.active_intent is not None:
            return bb.active_intent.intent not in {IntentType.STOP, IntentType.RETURN_TO_DOCK}
        return False

    def action_execute_voice_intent(self, bb: Blackboard) -> NodeStatus:
        if bb.pending_intent is not None:
            bb.active_intent = bb.pending_intent
            bb.pending_intent = None
            bb.planned_path = []
            bb.active_goal = None

        intent = bb.active_intent
        if intent is None:
            return NodeStatus.FAILURE

        if intent.intent == IntentType.GOTO_POSE and intent.target_pose is not None:
            bb.active_branch = "VOICE_GOTO"
            self.hal.set_eye_color(0.58, 1.0)
            status = self._plan_and_drive(bb, intent.target_pose, "VOICE_GOTO", arrival_tol_m=0.09)
            if status == NodeStatus.SUCCESS:
                pos = intent.target_pose
                self.hal.say_text(f"Arrived at target {pos.x:.1f}, {pos.y:.1f}.")
                bb.log_event(f"INTENT_EXEC:Arrived at ({pos.x:.2f}, {pos.y:.2f})")
                bb.active_intent = None
            return status

        if intent.intent == IntentType.GREET_PERSON:
            bb.active_branch = "VOICE_GREET"
            self.hal.stop_motors()
            self.hal.set_head_angle(0.36)
            self.hal.set_eye_color(0.48, 1.0)
            who = intent.person_name or "friend"
            self.hal.say_text(f"Hello {who}! Ready to explore.")
            bb.log_event(f"INTENT_EXEC:Greeted {who}")
            bb.active_intent = None
            return NodeStatus.SUCCESS

        if intent.intent == IntentType.STATUS_REPORT:
            bb.active_branch = "VOICE_STATUS"
            self.hal.stop_motors()
            pct = bb.telemetry.battery_pct if bb.telemetry else 0.0
            self.hal.say_text(f"Battery is at {pct:.0f} percent. Autonomy nominal.")
            bb.log_event(f"INTENT_EXEC:Status reported ({pct:.1f}%)")
            bb.active_intent = None
            return NodeStatus.SUCCESS

        if intent.intent == IntentType.PATROL:
            bb.active_intent = None
            bb.log_event("INTENT_EXEC:Resuming autonomous patrol")
            return NodeStatus.FAILURE  # Hand off to Patrol branch

        bb.active_intent = None
        return NodeStatus.FAILURE

    # --- 6. Social Face Greeting ---
    @staticmethod
    def is_face_ready_to_greet(bb: Blackboard) -> bool:
        if not bb.detected_faces:
            return False
        return bb.sim_time_s >= bb.face_greet_cooldown_until_s

    def action_greet_face(self, bb: Blackboard) -> NodeStatus:
        bb.active_branch = "FACE_GREET"
        face = bb.detected_faces[0]
        self.hal.stop_motors()
        self.hal.set_head_angle(0.38)
        self.hal.set_eye_color(0.50, 1.0)
        if abs(face.bearing_rad) > 0.14:
            omega = float(np.clip(2.5 * face.bearing_rad, -2.0, 2.0))
            self.hal.drive_twist(0.0, omega)
            return NodeStatus.RUNNING

        self.hal.say_text(f"Hi {face.name}! Nice to see you.")
        bb.face_greet_cooldown_until_s = bb.sim_time_s + 14.0
        bb.log_event(f"SOCIAL:Greeted human '{face.name}' at {face.distance_m:.2f}m")
        return NodeStatus.SUCCESS

    # --- 7. Autonomous Patrol ---
    def action_patrol_arena(self, bb: Blackboard) -> NodeStatus:
        bb.active_branch = "PATROL"
        self.hal.set_head_angle(0.08)
        self.hal.set_eye_color(0.45, 0.9)
        if not bb.patrol_waypoints:
            self.hal.stop_motors()
            return NodeStatus.SUCCESS

        target = bb.patrol_waypoints[bb.patrol_index % len(bb.patrol_waypoints)]
        status = self._plan_and_drive(bb, target, "PATROL", arrival_tol_m=0.10)
        if status == NodeStatus.SUCCESS:
            bb.log_event(
                f"PATROL:Reached waypoint {bb.patrol_index % len(bb.patrol_waypoints)} "
                f"({target.x:.2f}, {target.y:.2f})"
            )
            bb.patrol_index = (bb.patrol_index + 1) % len(bb.patrol_waypoints)
            return NodeStatus.RUNNING
        return NodeStatus.RUNNING


def build_vector_autonomy_tree(
    hal: VectorRobotHAL,
    planner: AStarPlanner | None = None,
) -> BTNode:
    """Construct the prioritized reactive Behavior Tree for Anki / DDL Vector."""
    ctrl = VectorAutonomyController(hal=hal, planner=planner)

    cliff_branch = Sequence(
        "CliffSafetyGuard",
        [
            Condition("CliffDetected", ctrl.is_cliff_active),
            Action("CliffRecover", ctrl.action_cliff_recover),
        ],
    )
    stop_branch = Sequence(
        "OperatorHaltGuard",
        [
            Condition("IsOperatorHalted", ctrl.is_operator_halted),
            Action("HoldStop", ctrl.action_hold_stop),
        ],
    )
    obstacle_branch = Sequence(
        "ObstacleSafetyGuard",
        [
            Condition("ObstacleTooClose", ctrl.is_obstacle_too_close),
            Action("ObstacleAvoid", ctrl.action_avoid_obstacle),
        ],
    )
    dock_branch = Sequence(
        "EnergyDockCharge",
        [
            Condition("NeedsDockCharge", ctrl.needs_dock_charge),
            Action("DockAndCharge", ctrl.action_dock_and_charge),
        ],
    )
    intent_branch = Sequence(
        "VoiceIntentDispatch",
        [
            Condition("HasVoiceIntent", ctrl.has_voice_intent),
            Action("ExecuteVoiceIntent", ctrl.action_execute_voice_intent),
        ],
    )
    social_branch = Sequence(
        "SocialFaceGreeting",
        [
            Condition("FaceReadyToGreet", ctrl.is_face_ready_to_greet),
            Action("GreetHumanFace", ctrl.action_greet_face),
        ],
    )
    patrol_branch = Action("AutonomousPatrol", ctrl.action_patrol_arena)

    return Selector(
        "VectorAutonomyRoot",
        [
            cliff_branch,
            stop_branch,
            obstacle_branch,
            dock_branch,
            intent_branch,
            social_branch,
            patrol_branch,
        ],
    )


def default_patrol_waypoints() -> list[Pose2D]:
    """Return canonical clockwise inspection loop waypoints in the 2.4m x 1.8m lab arena."""
    return [
        Pose2D(0.42, 1.05, math.pi / 2),
        Pose2D(1.12, 1.02, 0.0),
        Pose2D(1.18, 0.42, -math.pi / 2),
        Pose2D(1.98, 0.85, math.pi / 2),
        Pose2D(2.05, 1.48, math.pi),
        Pose2D(1.12, 1.54, math.pi),
    ]
