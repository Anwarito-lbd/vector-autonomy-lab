"""Deterministic simulated Anki / DDL Vector backend for hardware-free autonomy execution."""

from __future__ import annotations

import math

import numpy as np

from vector_lab.hal.base import HardwareConnectionError, VectorRobotHAL
from vector_lab.sim.environment import Arena2D
from vector_lab.sim.kinematics import (
    DifferentialDriveKinematics,
    DifferentialDriveParams,
    OdometryNoiseParams,
)
from vector_lab.types import (
    CameraFrame,
    Pose2D,
    RangeScan,
    TelemetryFrame,
    Twist2D,
    WheelVelocities,
)


class SimulatedVectorRobot(VectorRobotHAL):
    """Deterministic physics, sensor, battery, and camera simulator implementing VectorRobotHAL."""

    def __init__(
        self,
        arena: Arena2D | None = None,
        initial_pose: Pose2D | None = None,
        initial_battery_pct: float = 82.0,
        kinematics_params: DifferentialDriveParams | None = None,
        odom_noise: OdometryNoiseParams | None = None,
        scan_rays: int = 72,
        scan_max_range_m: float = 2.0,
        scan_noise_std_m: float = 0.004,
        seed: int = 42,
    ) -> None:
        self.arena = arena or Arena2D.default_lab_arena()
        self.kinematics = DifferentialDriveKinematics(kinematics_params)
        self.odom_noise = odom_noise or OdometryNoiseParams.nominal()
        self.scan_rays = scan_rays
        self.scan_max_range_m = scan_max_range_m
        self.scan_noise_std_m = scan_noise_std_m
        self._rng = np.random.default_rng(seed)

        start = initial_pose or Pose2D(0.36, 0.28, 0.0)
        self.ground_truth_pose: Pose2D = start
        self.odom_pose: Pose2D = start
        self.battery_pct: float = float(np.clip(initial_battery_pct, 0.0, 100.0))
        self.sim_time_s: float = 0.0
        self.head_angle_rad: float = 0.0
        self.lift_height_norm: float = 0.0
        self.eye_hue: float = 0.45
        self.eye_saturation: float = 1.0
        self.spoken_history: list[tuple[float, str]] = []
        self.trajectory_gt: list[Pose2D] = [start]
        self.trajectory_odom: list[Pose2D] = [start]

        self._connected: bool = False
        self._wheel_cmd = WheelVelocities(0.0, 0.0)
        self._last_collision: bool = False
        self._latest_scan: RangeScan | None = None
        self._latest_camera: CameraFrame | None = None

    def connect(self) -> None:
        self._connected = True
        self._refresh_sensors()

    def disconnect(self) -> None:
        self.stop_motors()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _require_connected(self) -> None:
        if not self._connected:
            raise HardwareConnectionError(
                "SimulatedVectorRobot is not connected; call connect() first."
            )

    def set_wheel_speeds(self, left_mps: float, right_mps: float) -> None:
        self._require_connected()
        max_w = self.kinematics.params.max_wheel_speed_mps
        self._wheel_cmd = WheelVelocities(
            v_left=float(np.clip(left_mps, -max_w, max_w)),
            v_right=float(np.clip(right_mps, -max_w, max_w)),
        )

    def drive_twist(self, linear_mps: float, angular_rps: float) -> None:
        self._require_connected()
        self._wheel_cmd = self.kinematics.twist_to_wheels(Twist2D(v=linear_mps, omega=angular_rps))

    def stop_motors(self) -> None:
        self._wheel_cmd = WheelVelocities(0.0, 0.0)

    def set_head_angle(self, angle_rad: float) -> None:
        self._require_connected()
        self.head_angle_rad = float(np.clip(angle_rad, math.radians(-22.0), math.radians(45.0)))

    def set_lift_height(self, height_norm: float) -> None:
        self._require_connected()
        self.lift_height_norm = float(np.clip(height_norm, 0.0, 1.0))

    def set_eye_color(self, hue: float, saturation: float = 1.0) -> None:
        self._require_connected()
        self.eye_hue = float(np.clip(hue, 0.0, 1.0))
        self.eye_saturation = float(np.clip(saturation, 0.0, 1.0))

    def say_text(self, text: str) -> None:
        self._require_connected()
        self.spoken_history.append((round(self.sim_time_s, 3), text))

    def _on_charger(self) -> bool:
        return self.ground_truth_pose.distance_to(self.arena.dock_pose) <= 0.12

    def _refresh_sensors(self) -> None:
        self._latest_scan = self.arena.cast_range_scan(
            pose=self.ground_truth_pose,
            timestamp_s=self.sim_time_s,
            num_rays=self.scan_rays,
            fov_rad=2.0 * math.pi,
            max_range_m=self.scan_max_range_m,
            range_noise_std_m=self.scan_noise_std_m,
            rng=self._rng,
        )
        self._latest_camera = self.arena.render_synthetic_camera(
            pose=self.ground_truth_pose,
            timestamp_s=self.sim_time_s,
        )

    def step(self, dt_s: float) -> None:
        """Advance exact kinematics, collisions, noisy odometry, battery, and sensors by dt_s."""
        self._require_connected()
        if dt_s <= 0.0:
            return

        candidate = self.kinematics.step_wheels(self.ground_truth_pose, self._wheel_cmd, dt_s)
        resolved, collided = self.arena.resolve_motion(
            self.ground_truth_pose,
            candidate,
            radius_m=self.kinematics.params.robot_radius_m,
        )
        self._last_collision = collided
        self.ground_truth_pose = resolved

        effective_wheels = (
            WheelVelocities(0.0, 0.0)
            if collided
            else self._wheel_cmd
        )
        self.odom_pose = self.kinematics.step_noisy_odometry(
            self.odom_pose,
            effective_wheels,
            dt_s,
            self.odom_noise,
            self._rng,
        )

        self.sim_time_s += dt_s
        on_dock = self._on_charger()
        if on_dock:
            self.battery_pct = min(100.0, self.battery_pct + 4.5 * dt_s)
        else:
            motion_load = 0.35 * (abs(self._wheel_cmd.v_left) + abs(self._wheel_cmd.v_right))
            self.battery_pct = max(0.0, self.battery_pct - (0.28 + motion_load) * dt_s)

        self.trajectory_gt.append(self.ground_truth_pose)
        self.trajectory_odom.append(self.odom_pose)
        self._refresh_sensors()

    def read_telemetry(self) -> TelemetryFrame:
        self._require_connected()
        assert self._latest_scan is not None
        on_dock = self._on_charger()
        cliff = self.arena.check_cliffs(self.ground_truth_pose)
        twist = self.kinematics.wheels_to_twist(self._wheel_cmd)
        # LiPo single-cell voltage curve [3.50V .. 4.20V]
        voltage = 3.50 + 0.70 * (self.battery_pct / 100.0)
        return TelemetryFrame(
            timestamp_s=self.sim_time_s,
            pose_odom=self.odom_pose,
            wheel_speeds=self._wheel_cmd,
            battery_voltage=round(voltage, 3),
            battery_pct=round(self.battery_pct, 2),
            is_charging=on_dock and self.battery_pct < 99.9,
            on_charger=on_dock,
            cliff=cliff,
            proximity_mm=round(self._latest_scan.min_forward_range_m * 1000.0, 1),
            head_angle_rad=self.head_angle_rad,
            lift_height_norm=self.lift_height_norm,
            imu_accel_mps2=(0.0, 0.0, 9.81),
            imu_gyro_rps=(0.0, 0.0, twist.omega),
        )

    def read_camera_frame(self) -> CameraFrame:
        self._require_connected()
        assert self._latest_camera is not None
        return self._latest_camera

    def read_range_scan(self) -> RangeScan:
        self._require_connected()
        assert self._latest_scan is not None
        return self._latest_scan
