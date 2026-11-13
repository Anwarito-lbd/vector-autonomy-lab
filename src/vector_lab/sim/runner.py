"""Closed-loop multi-episode deterministic autonomy & SLAM simulation runner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from vector_lab.hal.simulated import SimulatedVectorRobot
from vector_lab.perception.intent import VoiceIntentPipeline
from vector_lab.perception.vision import VisionProcessor
from vector_lab.sim.environment import Arena2D
from vector_lab.sim.kinematics import OdometryNoiseParams
from vector_lab.sim.planner import AStarPlanner
from vector_lab.sim.slam import OccupancyGridConfig, OccupancyGridSLAM
from vector_lab.types import Blackboard, Pose2D, wrap_angle


@dataclass(slots=True)
class TickRecord:
    """Single-tick closed-loop telemetry and autonomy trace."""

    time_s: float
    gt_pose: Pose2D
    odom_pose: Pose2D
    slam_pose: Pose2D
    battery_pct: float
    min_forward_range_m: float
    linear_v_mps: float
    angular_omega_rps: float
    active_branch: str
    odom_pos_err_m: float
    slam_pos_err_m: float
    odom_heading_err_rad: float
    slam_heading_err_rad: float


@dataclass(slots=True)
class EpisodeResult:
    """Complete deterministic episode output including trajectories, SLAM grid, and metrics."""

    name: str
    records: list[TickRecord]
    slam: OccupancyGridSLAM
    arena: Arena2D
    blackboard: Blackboard
    slam_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def mean_odom_error_m(self) -> float:
        return sum(r.odom_pos_err_m for r in self.records) / max(1, len(self.records))

    @property
    def max_odom_error_m(self) -> float:
        return max((r.odom_pos_err_m for r in self.records), default=0.0)

    @property
    def mean_slam_error_m(self) -> float:
        return sum(r.slam_pos_err_m for r in self.records) / max(1, len(self.records))

    @property
    def max_slam_error_m(self) -> float:
        return max((r.slam_pos_err_m for r in self.records), default=0.0)


class SimulationEpisodeRunner:
    """Runs deterministic closed-loop HAL + Perception + SLAM + Behavior-Tree episodes."""

    def __init__(
        self,
        arena: Arena2D | None = None,
        dt_s: float = 0.1,
    ) -> None:
        self.arena = arena or Arena2D.default_lab_arena()
        self.dt_s = dt_s

    def run_episode(
        self,
        name: str,
        *,
        num_steps: int = 380,
        initial_pose: Pose2D | None = None,
        initial_battery_pct: float = 42.0,
        low_battery_threshold_pct: float = 25.0,
        odom_noise: OdometryNoiseParams | None = None,
        use_scan_matching: bool = True,
        scheduled_intents: dict[int, str] | None = None,
        patrol_waypoints: list[Pose2D] | None = None,
        seed: int = 42,
    ) -> EpisodeResult:
        """Execute a deterministic autonomy episode and collect full metrics."""
        from vector_lab.behavior_tree.autonomy import (
            build_vector_autonomy_tree,
            default_patrol_waypoints,
        )

        robot = SimulatedVectorRobot(
            arena=self.arena,
            initial_pose=initial_pose or Pose2D(0.40, 0.30, 0.0),
            initial_battery_pct=initial_battery_pct,
            odom_noise=odom_noise or OdometryNoiseParams.nominal(),
            seed=seed,
        )
        robot.connect()

        slam = OccupancyGridSLAM(
            OccupancyGridConfig(
                width_m=self.arena.width_m,
                height_m=self.arena.height_m,
                resolution_m=0.025,
            )
        )
        vision = VisionProcessor()
        intent_pipeline = VoiceIntentPipeline()
        planner = AStarPlanner(inflation_radius_m=0.065)
        tree = build_vector_autonomy_tree(robot, planner=planner)

        bb = Blackboard(
            slam_map=slam,
            estimated_pose=robot.ground_truth_pose,
            patrol_waypoints=patrol_waypoints or default_patrol_waypoints(),
            dock_approach_pose=self.arena.dock_approach_pose,
            dock_pose=self.arena.dock_pose,
            low_battery_threshold_pct=low_battery_threshold_pct,
            charged_threshold_pct=92.0,
        )

        # Maintain incremental odometry delta so scan-matched SLAM pose corrects cumulative drift
        prev_odom = robot.odom_pose
        slam_pose = robot.ground_truth_pose
        records: list[TickRecord] = []
        scheduled = scheduled_intents or {}

        for step_idx in range(num_steps):
            bb.tick_count = step_idx
            bb.sim_time_s = robot.sim_time_s

            if step_idx in scheduled:
                intent_pipeline.dispatch(scheduled[step_idx], bb)

            telemetry = robot.read_telemetry()
            camera = robot.read_camera_frame()
            scan = robot.read_range_scan()

            # Propagate SLAM pose by odometry increment, then refine via correlative scan matching
            dx_odom = telemetry.pose_odom.x - prev_odom.x
            dy_odom = telemetry.pose_odom.y - prev_odom.y
            dth_odom = wrap_angle(telemetry.pose_odom.theta - prev_odom.theta)
            # Rotate local displacement from odom frame into slam_pose frame
            rot_err = wrap_angle(slam_pose.theta - prev_odom.theta)
            cos_r, sin_r = math.cos(rot_err), math.sin(rot_err)
            pred_pose = Pose2D(
                x=slam_pose.x + dx_odom * cos_r - dy_odom * sin_r,
                y=slam_pose.y + dx_odom * sin_r + dy_odom * cos_r,
                theta=wrap_angle(slam_pose.theta + dth_odom),
            )
            prev_odom = telemetry.pose_odom

            if use_scan_matching and step_idx >= 3:
                slam_pose = slam.correlative_scan_match(
                    pred_pose,
                    scan,
                    xy_window_m=0.04,
                    xy_step_m=0.02,
                    theta_window_rad=0.06,
                    theta_step_rad=0.02,
                )
            else:
                slam_pose = pred_pose

            bb.telemetry = telemetry
            bb.estimated_pose = slam_pose
            slam.update(slam_pose, scan)
            vision.process(camera, scan, bb)

            tree.tick(bb)

            gt = robot.ground_truth_pose
            twist = robot.kinematics.wheels_to_twist(telemetry.wheel_speeds)
            records.append(
                TickRecord(
                    time_s=round(robot.sim_time_s, 3),
                    gt_pose=gt,
                    odom_pose=telemetry.pose_odom,
                    slam_pose=slam_pose,
                    battery_pct=telemetry.battery_pct,
                    min_forward_range_m=scan.min_forward_range_m,
                    linear_v_mps=twist.v,
                    angular_omega_rps=twist.omega,
                    active_branch=bb.active_branch,
                    odom_pos_err_m=gt.distance_to(telemetry.pose_odom),
                    slam_pos_err_m=gt.distance_to(slam_pose),
                    odom_heading_err_rad=abs(wrap_angle(gt.theta - telemetry.pose_odom.theta)),
                    slam_heading_err_rad=abs(wrap_angle(gt.theta - slam_pose.theta)),
                )
            )

            robot.step(self.dt_s)

        robot.disconnect()
        slam_metrics = slam.evaluate_against_ground_truth(self.arena)
        return EpisodeResult(
            name=name,
            records=records,
            slam=slam,
            arena=self.arena,
            blackboard=bb,
            slam_metrics=slam_metrics,
        )
