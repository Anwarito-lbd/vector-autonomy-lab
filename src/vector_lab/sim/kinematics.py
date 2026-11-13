"""Differential-drive unicycle kinematics, wheel conversion, and noisy odometry model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vector_lab.types import Pose2D, Twist2D, WheelVelocities, wrap_angle


@dataclass(frozen=True, slots=True)
class DifferentialDriveParams:
    """Physical geometry and actuator limits modeled on Anki / DDL Vector."""

    track_width_m: float = 0.048
    wheel_radius_m: float = 0.014
    robot_radius_m: float = 0.045
    max_wheel_speed_mps: float = 0.22
    max_linear_speed_mps: float = 0.20
    max_angular_speed_rps: float = 3.20


@dataclass(frozen=True, slots=True)
class OdometryNoiseParams:
    """Systematic tread scale drift and stochastic wheel-slip Gaussian parameters."""

    left_scale_bias: float = 1.0
    right_scale_bias: float = 1.0
    linear_noise_std_mps: float = 0.0
    angular_noise_std_rps: float = 0.0

    @classmethod
    def nominal(cls) -> OdometryNoiseParams:
        """Return realistic tabletop tread-slip calibration parameters."""
        return cls(
            left_scale_bias=1.015,
            right_scale_bias=0.988,
            linear_noise_std_mps=0.004,
            angular_noise_std_rps=0.025,
        )

    @classmethod
    def high_slip(cls) -> OdometryNoiseParams:
        """Return stressed high-slip carpet/glossy-table noise parameters."""
        return cls(
            left_scale_bias=1.040,
            right_scale_bias=0.965,
            linear_noise_std_mps=0.012,
            angular_noise_std_rps=0.075,
        )


class DifferentialDriveKinematics:
    """Exact ICC arc integrator and wheel-encoder odometry simulator."""

    def __init__(self, params: DifferentialDriveParams | None = None) -> None:
        self.params = params or DifferentialDriveParams()

    def twist_to_wheels(self, twist: Twist2D) -> WheelVelocities:
        """Convert (v, omega) twist to left/right tread speeds with curvature-preserving clamp."""
        p = self.params
        v = float(np.clip(twist.v, -p.max_linear_speed_mps, p.max_linear_speed_mps))
        omega = float(np.clip(twist.omega, -p.max_angular_speed_rps, p.max_angular_speed_rps))
        half_L = 0.5 * p.track_width_m
        v_left = v - half_L * omega
        v_right = v + half_L * omega

        peak = max(abs(v_left), abs(v_right))
        if peak > self.params.max_wheel_speed_mps:
            scale = self.params.max_wheel_speed_mps / peak
            v_left *= scale
            v_right *= scale
        return WheelVelocities(v_left=v_left, v_right=v_right)

    def wheels_to_twist(self, wheels: WheelVelocities) -> Twist2D:
        """Convert left/right tread linear velocities (m/s) back into body twist (v, omega)."""
        v = 0.5 * (wheels.v_left + wheels.v_right)
        omega = (wheels.v_right - wheels.v_left) / self.params.track_width_m
        return Twist2D(v=v, omega=omega)

    @staticmethod
    def step_pose(pose: Pose2D, twist: Twist2D, dt_s: float) -> Pose2D:
        """Advance SE(2) pose over dt_s using exact Instantaneous Center of Curvature (ICC) arc."""
        if dt_s <= 0.0:
            return pose
        v = twist.v
        omega = twist.omega
        if abs(omega) < 1e-7:
            mid_theta = pose.theta + 0.5 * omega * dt_s
            nx = pose.x + v * math.cos(mid_theta) * dt_s
            ny = pose.y + v * math.sin(mid_theta) * dt_s
            ntheta = wrap_angle(pose.theta + omega * dt_s)
            return Pose2D(x=nx, y=ny, theta=ntheta)

        radius = v / omega
        dtheta = omega * dt_s
        nx = pose.x + radius * (math.sin(pose.theta + dtheta) - math.sin(pose.theta))
        ny = pose.y - radius * (math.cos(pose.theta + dtheta) - math.cos(pose.theta))
        ntheta = wrap_angle(pose.theta + dtheta)
        return Pose2D(x=nx, y=ny, theta=ntheta)

    def step_wheels(self, pose: Pose2D, wheels: WheelVelocities, dt_s: float) -> Pose2D:
        """Advance SE(2) pose directly from left/right tread speeds."""
        return self.step_pose(pose, self.wheels_to_twist(wheels), dt_s)

    def step_noisy_odometry(
        self,
        odom_pose: Pose2D,
        commanded_wheels: WheelVelocities,
        dt_s: float,
        noise: OdometryNoiseParams,
        rng: np.random.Generator,
    ) -> Pose2D:
        """Integrate wheel odometry corrupted by tread-scale bias and Gaussian slip noise."""
        biased_wheels = WheelVelocities(
            v_left=commanded_wheels.v_left * noise.left_scale_bias,
            v_right=commanded_wheels.v_right * noise.right_scale_bias,
        )
        twist = self.wheels_to_twist(biased_wheels)
        noisy_v = twist.v + (
            float(rng.normal(0.0, noise.linear_noise_std_mps))
            if abs(twist.v) > 1e-6 and noise.linear_noise_std_mps > 0.0
            else 0.0
        )
        noisy_omega = twist.omega + (
            float(rng.normal(0.0, noise.angular_noise_std_rps))
            if (abs(twist.v) > 1e-6 or abs(twist.omega) > 1e-6)
            and noise.angular_noise_std_rps > 0.0
            else 0.0
        )
        return self.step_pose(odom_pose, Twist2D(v=noisy_v, omega=noisy_omega), dt_s)
