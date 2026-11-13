"""Unit tests for differential-drive kinematics, ICC arc integration, and noisy odometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vector_lab.sim.kinematics import (
    DifferentialDriveKinematics,
    OdometryNoiseParams,
)
from vector_lab.types import Pose2D, Twist2D, WheelVelocities


def test_twist_wheel_roundtrip_and_curvature_preserving_clamp() -> None:
    kin = DifferentialDriveKinematics()
    twist = Twist2D(v=0.14, omega=1.2)
    wheels = kin.twist_to_wheels(twist)
    recovered = kin.wheels_to_twist(wheels)
    assert recovered.v == pytest.approx(twist.v, abs=1e-9)
    assert recovered.omega == pytest.approx(twist.omega, abs=1e-9)

    # Excessive command saturates wheel speed while preserving v/omega turning curvature
    fast_twist = Twist2D(v=0.20, omega=2.5)
    clamped_wheels = kin.twist_to_wheels(fast_twist)
    max_w = kin.params.max_wheel_speed_mps + 1e-9
    assert max(abs(clamped_wheels.v_left), abs(clamped_wheels.v_right)) <= max_w
    clamped_twist = kin.wheels_to_twist(clamped_wheels)
    assert clamped_twist.v / clamped_twist.omega == pytest.approx(
        fast_twist.v / fast_twist.omega, rel=1e-6
    )


def test_icc_arc_full_circle_closure_invariant() -> None:
    kin = DifferentialDriveKinematics()
    pose = Pose2D(x=1.0, y=0.5, theta=0.0)
    v = 0.12
    omega = 1.0
    steps = 120
    dt = (2.0 * math.pi / omega) / steps
    for _ in range(steps):
        pose = kin.step_pose(pose, Twist2D(v=v, omega=omega), dt)

    # Full 2*pi circle must return to starting coordinates within floating-point tolerance
    assert pose.x == pytest.approx(1.0, abs=1e-9)
    assert pose.y == pytest.approx(0.5, abs=1e-9)
    assert abs(pose.theta) < 1e-9


def test_straight_line_motion_preserves_heading() -> None:
    kin = DifferentialDriveKinematics()
    pose = Pose2D(0.2, 0.3, math.pi / 4.0)
    nxt = kin.step_pose(pose, Twist2D(v=0.20, omega=0.0), dt_s=1.0)
    assert nxt.x == pytest.approx(0.2 + 0.20 * math.cos(math.pi / 4.0), abs=1e-9)
    assert nxt.y == pytest.approx(0.3 + 0.20 * math.sin(math.pi / 4.0), abs=1e-9)
    assert nxt.theta == pytest.approx(math.pi / 4.0, abs=1e-9)


def test_noisy_odometry_deterministic_with_seed_and_reflects_bias() -> None:
    kin = DifferentialDriveKinematics()
    wheels = WheelVelocities(0.15, 0.15)
    noise = OdometryNoiseParams.nominal()

    p1 = Pose2D(0.0, 0.0, 0.0)
    p2 = Pose2D(0.0, 0.0, 0.0)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    for _ in range(40):
        p1 = kin.step_noisy_odometry(p1, wheels, 0.1, noise, rng1)
        p2 = kin.step_noisy_odometry(p2, wheels, 0.1, noise, rng2)

    assert p1 == p2
    # Systematic left_scale_bias > right_scale_bias induces clockwise negative heading drift
    assert p1.theta < -0.05
