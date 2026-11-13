"""End-to-end closed-loop deterministic autonomy & SLAM integration tests."""

from __future__ import annotations

from vector_lab.sim.runner import SimulationEpisodeRunner
from vector_lab.types import Pose2D


def test_closed_loop_patrol_voice_goto_face_greet_and_dock_charge() -> None:
    runner = SimulationEpisodeRunner(dt_s=0.1)
    result = runner.run_episode(
        "integration_full",
        num_steps=560,
        initial_pose=Pose2D(0.42, 0.30, 0.0),
        initial_battery_pct=36.0,
        low_battery_threshold_pct=25.0,
        scheduled_intents={
            90: "Navigate to x=1.14, y=1.36",
        },
        seed=42,
    )

    branches_seen = {r.active_branch for r in result.records}
    assert "PATROL" in branches_seen
    assert "VOICE_GOTO" in branches_seen
    assert "FACE_GREET" in branches_seen
    assert "DOCK_CHARGE" in branches_seen
    assert result.slam_metrics["coverage_pct"] >= 60.0
    assert result.slam_metrics["occupied_precision"] >= 0.70
    # Correlative scan matching must keep mean SLAM error tighter than raw odometry drift
    assert result.mean_slam_error_m <= result.mean_odom_error_m + 0.01
    # Robot should reach charger and begin recharging before episode end
    recharged = any(
        r.battery_pct > result.records[-50].battery_pct for r in result.records[-30:]
    )
    logged_charge = any("Charging on dock" in msg for _, msg in result.blackboard.event_log)
    assert recharged or logged_charge
