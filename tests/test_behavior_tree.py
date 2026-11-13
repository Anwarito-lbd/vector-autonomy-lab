"""Tests for Behavior-Tree composites, reactive preemption, and autonomy branches."""

from __future__ import annotations

from vector_lab.behavior_tree import (
    Action,
    Condition,
    Inverter,
    NodeStatus,
    Parallel,
    Selector,
    Sequence,
    build_vector_autonomy_tree,
    default_patrol_waypoints,
)
from vector_lab.hal.simulated import SimulatedVectorRobot
from vector_lab.sim.slam import OccupancyGridSLAM
from vector_lab.types import (
    Blackboard,
    CliffSensors,
    DetectedFace,
    IntentGoal,
    IntentType,
    Pose2D,
    TelemetryFrame,
    WheelVelocities,
)


def test_bt_composites_parallel_inverter_and_reactive_preemption() -> None:
    bb = Blackboard()
    bb.metadata["danger"] = False
    trace: list[str] = []

    def danger_action(_: Blackboard) -> NodeStatus:
        trace.append("danger")
        return NodeStatus.SUCCESS

    def work_action(_: Blackboard) -> NodeStatus:
        trace.append("work")
        return NodeStatus.RUNNING

    root = Selector(
        "root",
        [
            Sequence(
                "safety",
                [
                    Condition("is_danger", lambda b: bool(b.metadata["danger"])),
                    Action("handle_danger", danger_action),
                ],
            ),
            Action("do_work", work_action),
        ],
    )

    assert root.tick(bb) == NodeStatus.RUNNING
    assert trace == ["work"]

    # High-priority danger must preempt RUNNING work on the very next tick
    bb.metadata["danger"] = True
    assert root.tick(bb) == NodeStatus.SUCCESS
    assert trace == ["work", "danger"]

    inv = Inverter("inv", Condition("false_cond", lambda _: False))
    assert inv.tick(bb) == NodeStatus.SUCCESS

    par = Parallel(
        "par",
        [
            Action("a1", lambda _: NodeStatus.SUCCESS),
            Action("a2", lambda _: NodeStatus.RUNNING),
        ],
        success_threshold=1,
        failure_threshold=1,
    )
    assert par.tick(bb) == NodeStatus.SUCCESS


def _make_telemetry(
    *,
    battery_pct: float = 80.0,
    on_charger: bool = False,
    cliff: CliffSensors | None = None,
    pose: Pose2D | None = None,
) -> TelemetryFrame:
    return TelemetryFrame(
        timestamp_s=1.0,
        pose_odom=pose or Pose2D(0.5, 0.5, 0.0),
        wheel_speeds=WheelVelocities(0.0, 0.0),
        battery_voltage=4.0,
        battery_pct=battery_pct,
        is_charging=on_charger,
        on_charger=on_charger,
        cliff=cliff or CliffSensors(),
        proximity_mm=800.0,
        head_angle_rad=0.0,
        lift_height_norm=0.0,
    )


def test_autonomy_tree_preempts_for_cliff_stop_obstacle_battery_and_face() -> None:
    robot = SimulatedVectorRobot()
    robot.connect()
    tree = build_vector_autonomy_tree(robot)

    bb = Blackboard(
        slam_map=OccupancyGridSLAM(),
        estimated_pose=Pose2D(0.5, 0.5, 0.0),
        patrol_waypoints=default_patrol_waypoints(),
    )

    # 1. Nominal patrol
    bb.telemetry = _make_telemetry(battery_pct=80.0)
    bb.forward_obstacle_dist_m = 0.90
    tree.tick(bb)
    assert bb.active_branch == "PATROL"

    # 2. Face detected triggers social greeting
    bb.detected_faces = [
        DetectedFace(
            face_id=1,
            name="Anwar",
            bbox_xywh=(60, 40, 24, 28),
            distance_m=0.45,
            bearing_rad=0.02,
            confidence=0.95,
        )
    ]
    status = tree.tick(bb)
    assert bb.active_branch == "FACE_GREET"
    assert status == NodeStatus.SUCCESS
    assert any("Hi Anwar!" in msg for _, msg in robot.spoken_history)

    # 3. Low battery preempts patrol for dock charge
    bb.detected_faces = []
    bb.telemetry = _make_telemetry(battery_pct=18.0)
    tree.tick(bb)
    assert bb.active_branch == "DOCK_CHARGE"

    # 4. Obstacle too close preempts dock navigation
    bb.forward_obstacle_dist_m = 0.08
    tree.tick(bb)
    assert bb.active_branch == "OBSTACLE_AVOID"

    # 5. Operator STOP preempts obstacle avoidance
    bb.pending_intent = IntentGoal(intent=IntentType.STOP, utterance="stop")
    tree.tick(bb)
    assert bb.active_branch == "OPERATOR_STOP"

    # 6. Cliff detection holds top safety priority even over operator stop
    bb.telemetry = _make_telemetry(cliff=CliffSensors(front_left=True))
    tree.tick(bb)
    assert bb.active_branch == "CLIFF_AVOID"
    robot.disconnect()
