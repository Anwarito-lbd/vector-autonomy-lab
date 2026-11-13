"""Tests for HAL factory/adapters, SimulatedVectorRobot, CV perception, and voice intents."""

from __future__ import annotations

import pytest

from vector_lab.hal import (
    HardwareConnectionError,
    SimulatedVectorRobot,
    VectorConnectionConfig,
    WirePodGRPCAdapter,
    create_robot_hal,
)
from vector_lab.perception import (
    DeterministicIntentParser,
    LLMIntentAdapter,
    VisionProcessor,
    VoiceIntentPipeline,
)
from vector_lab.types import Blackboard, IntentType, Pose2D


def test_hal_factory_and_wirepod_payloads() -> None:
    sim = create_robot_hal(None)
    assert isinstance(sim, SimulatedVectorRobot)

    hw = create_robot_hal("00e12345", host="192.168.1.50")
    assert isinstance(hw, WirePodGRPCAdapter)
    assert isinstance(hw.config, VectorConnectionConfig)

    payload = WirePodGRPCAdapter.build_drive_wheels_payload(0.12, -0.06)
    assert payload["left_wheel_mmps"] == pytest.approx(120.0)
    assert payload["right_wheel_mmps"] == pytest.approx(-60.0)

    url = hw.build_wirepod_intent_url("intent_greeting_hello", "Anwar")
    assert "192.168.1.50:8080" in url
    assert "serial=00e12345" in url

    parsed_tel = WirePodGRPCAdapter.parse_sdk_state_dict(
        {
            "timestamp_s": 4.2,
            "pose": {"x_mm": 450.0, "y_mm": -120.0, "angle_rad": 0.5},
            "battery_voltage": 3.85,
            "is_charging": False,
            "on_charger": False,
            "prox_distance_mm": 310.0,
            "cliff_flags": {"front_right": True},
        }
    )
    assert parsed_tel.pose_odom.x == pytest.approx(0.45)
    assert parsed_tel.pose_odom.y == pytest.approx(-0.12)
    assert parsed_tel.cliff.front_right is True
    assert 40.0 <= parsed_tel.battery_pct <= 60.0


def test_simulated_robot_connection_guard_and_cliff_sensor() -> None:
    robot = SimulatedVectorRobot(initial_pose=Pose2D(2.15, 0.20, 0.0))
    with pytest.raises(HardwareConnectionError):
        robot.read_telemetry()

    robot.connect()
    assert robot.is_connected
    tel = robot.read_telemetry()
    # Inside SE table cutout cliff zone (x in [1.98, 2.34], y in [0.04, 0.34])
    assert tel.cliff.any_triggered is True

    cam = robot.read_camera_frame()
    scan = robot.read_range_scan()
    bb = Blackboard()
    report = VisionProcessor().process(cam, scan, bb)
    assert 0.0 <= report.visual_edge_density <= 1.0
    assert bb.forward_obstacle_dist_m == pytest.approx(scan.min_forward_range_m, abs=1e-4)
    robot.disconnect()


def test_deterministic_voice_intent_parser_and_llm_offline_fallback() -> None:
    parser = DeterministicIntentParser()
    llm_fallback = LLMIntentAdapter(endpoint_url=None, fallback=parser)
    pipeline = VoiceIntentPipeline(adapter=llm_fallback)
    bb = Blackboard()

    g_stop = pipeline.dispatch("Vector stop right now!", bb)
    assert g_stop.intent == IntentType.STOP

    g_dock = pipeline.dispatch("Go back to the charger dock", bb)
    assert g_dock.intent == IntentType.RETURN_TO_DOCK

    g_goto = pipeline.dispatch("Navigate to x=1.45, y=0.95", bb)
    assert g_goto.intent == IntentType.GOTO_POSE
    assert g_goto.target_pose is not None
    assert g_goto.target_pose.x == pytest.approx(1.45)
    assert g_goto.target_pose.y == pytest.approx(0.95)

    g_greet = pipeline.dispatch("Say hello to Anwar", bb)
    assert g_greet.intent == IntentType.GREET_PERSON
    assert g_greet.person_name == "Anwar"

    g_patrol = pipeline.dispatch("Resume autonomous patrol", bb)
    assert g_patrol.intent == IntentType.PATROL

    g_status = pipeline.dispatch("Give me a battery status report", bb)
    assert g_status.intent == IntentType.STATUS_REPORT
