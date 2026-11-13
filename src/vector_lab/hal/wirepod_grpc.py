"""Wire-Pod / DDL Vector SDK gRPC Hardware Abstraction Adapter."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import numpy as np

from vector_lab.hal.base import HardwareConnectionError, VectorRobotHAL
from vector_lab.sim.kinematics import DifferentialDriveKinematics
from vector_lab.types import (
    CameraFrame,
    CliffSensors,
    Pose2D,
    RangeScan,
    TelemetryFrame,
    Twist2D,
    WheelVelocities,
)


@dataclass(frozen=True, slots=True)
class VectorConnectionConfig:
    """Connection parameters for physical DDL Vector via Wire-Pod / official gRPC SDK."""

    serial: str
    host: str = "127.0.0.1"
    wirepod_port: int = 8080
    grpc_port: int = 443
    cert_path: str | None = None
    guid: str | None = None


class WirePodGRPCAdapter(VectorRobotHAL):
    """Hardware adapter translating VectorRobotHAL calls to DDL Vector gRPC / Wire-Pod APIs."""

    def __init__(self, config: VectorConnectionConfig) -> None:
        self.config = config
        self.kinematics = DifferentialDriveKinematics()
        self._robot: Any = None
        self._connected: bool = False

    @staticmethod
    def build_drive_wheels_payload(left_mps: float, right_mps: float) -> dict[str, float]:
        """Convert SI tread velocities (m/s) into DDL Vector SDK mm/s DriveWheels payload."""
        return {
            "left_wheel_mmps": round(left_mps * 1000.0, 2),
            "right_wheel_mmps": round(right_mps * 1000.0, 2),
            "left_wheel_accel_mmps2": 0.0,
            "right_wheel_accel_mmps2": 0.0,
        }

    def build_wirepod_intent_url(self, intent_name: str, param: str = "") -> str:
        """Build Wire-Pod REST API endpoint URL for intent injection or robot control."""
        query = urlencode({"serial": self.config.serial, "intent": intent_name, "param": param})
        return f"http://{self.config.host}:{self.config.wirepod_port}/api-sdk/cloud_intent?{query}"

    @staticmethod
    def parse_sdk_state_dict(raw: dict[str, Any]) -> TelemetryFrame:
        """Normalize raw Vector gRPC RobotState fields into a TelemetryFrame."""
        pose_dict = raw.get("pose", {})
        pose_odom = Pose2D(
            x=float(pose_dict.get("x_mm", 0.0)) / 1000.0,
            y=float(pose_dict.get("y_mm", 0.0)) / 1000.0,
            theta=float(pose_dict.get("angle_rad", 0.0)),
        )
        cliff_raw = raw.get("cliff_flags", {})
        cliff = CliffSensors(
            front_left=bool(cliff_raw.get("front_left", False)),
            front_right=bool(cliff_raw.get("front_right", False)),
            back_left=bool(cliff_raw.get("back_left", False)),
            back_right=bool(cliff_raw.get("back_right", False)),
        )
        voltage = float(raw.get("battery_voltage", 3.85))
        pct = float(np.clip((voltage - 3.50) / 0.70 * 100.0, 0.0, 100.0))
        return TelemetryFrame(
            timestamp_s=float(raw.get("timestamp_s", 0.0)),
            pose_odom=pose_odom,
            wheel_speeds=WheelVelocities(
                v_left=float(raw.get("left_wheel_speed_mmps", 0.0)) / 1000.0,
                v_right=float(raw.get("right_wheel_speed_mmps", 0.0)) / 1000.0,
            ),
            battery_voltage=voltage,
            battery_pct=round(pct, 2),
            is_charging=bool(raw.get("is_charging", False)),
            on_charger=bool(raw.get("on_charger", False)),
            cliff=cliff,
            proximity_mm=float(raw.get("prox_distance_mm", 400.0)),
            head_angle_rad=float(raw.get("head_angle_rad", 0.0)),
            lift_height_norm=float(raw.get("lift_height_norm", 0.0)),
        )

    def connect(self) -> None:
        try:
            anki_vector = importlib.import_module("anki_vector")
        except ImportError as exc:
            raise HardwareConnectionError(
                "Physical Vector SDK ('anki_vector') is not installed. "
                "Install optional '[hardware]' extras or omit target to run SimulatedVectorRobot."
            ) from exc

        try:
            robot = anki_vector.Robot(serial=self.config.serial)
            robot.connect()
            self._robot = robot
            self._connected = True
        except Exception as exc:  # pragma: no cover - requires physical robot hardware
            raise HardwareConnectionError(
                f"Failed to connect to Vector serial={self.config.serial} "
                f"at {self.config.host}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        if self._robot is not None:
            try:
                self._robot.motors.stop_all_motors()
                self._robot.disconnect()
            finally:
                self._robot = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _require_robot(self) -> Any:
        if not self._connected or self._robot is None:
            raise HardwareConnectionError(
                "WirePodGRPCAdapter is not connected to a physical Vector."
            )
        return self._robot

    def set_wheel_speeds(self, left_mps: float, right_mps: float) -> None:
        robot = self._require_robot()
        payload = self.build_drive_wheels_payload(left_mps, right_mps)
        robot.motors.set_wheel_motors(payload["left_wheel_mmps"], payload["right_wheel_mmps"])

    def drive_twist(self, linear_mps: float, angular_rps: float) -> None:
        wheels = self.kinematics.twist_to_wheels(Twist2D(v=linear_mps, omega=angular_rps))
        self.set_wheel_speeds(wheels.v_left, wheels.v_right)

    def stop_motors(self) -> None:
        robot = self._require_robot()
        robot.motors.stop_all_motors()

    def set_head_angle(self, angle_rad: float) -> None:
        robot = self._require_robot()
        anki_vector = importlib.import_module("anki_vector")
        robot.behavior.set_head_angle(anki_vector.util.radians(angle_rad))

    def set_lift_height(self, height_norm: float) -> None:
        robot = self._require_robot()
        robot.behavior.set_lift_height(float(np.clip(height_norm, 0.0, 1.0)))

    def set_eye_color(self, hue: float, saturation: float = 1.0) -> None:
        robot = self._require_robot()
        robot.behavior.set_eye_color(hue=hue, saturation=saturation)

    def say_text(self, text: str) -> None:
        robot = self._require_robot()
        robot.behavior.say_text(text)

    def read_telemetry(self) -> TelemetryFrame:
        robot = self._require_robot()
        prox = robot.proximity.last_sensor_reading
        raw = {
            "timestamp_s": 0.0,
            "pose": {
                "x_mm": robot.pose.position.x,
                "y_mm": robot.pose.position.y,
                "angle_rad": robot.pose.rotation.angle_z.radians,
            },
            "battery_voltage": robot.get_battery_state().battery_volts,
            "is_charging": robot.status.is_charging,
            "on_charger": robot.status.is_on_charger,
            "prox_distance_mm": prox.distance.distance_mm if prox else 500.0,
            "head_angle_rad": robot.head_angle_rad,
            "lift_height_norm": robot.lift_height_mm / 92.0,
            "cliff_flags": {"front_left": robot.status.is_cliff_detected},
        }
        return self.parse_sdk_state_dict(raw)

    def read_camera_frame(self) -> CameraFrame:
        robot = self._require_robot()
        img = np.asarray(robot.camera.latest_image.raw_image.convert("L"), dtype=np.uint8)
        return CameraFrame(
            timestamp_s=0.0,
            width=int(img.shape[1]),
            height=int(img.shape[0]),
            image=img,
            brightness=float(np.mean(img) / 255.0),
        )

    def read_range_scan(self) -> RangeScan:
        telemetry = self.read_telemetry()
        forward_m = min(2.0, max(0.03, telemetry.proximity_mm / 1000.0))
        return RangeScan(
            timestamp_s=telemetry.timestamp_s,
            angles_rad=np.array([0.0], dtype=np.float64),
            ranges_m=np.array([forward_m], dtype=np.float64),
            max_range_m=2.0,
            elevations_rad=np.array([math.radians(0.0)], dtype=np.float64),
        )
