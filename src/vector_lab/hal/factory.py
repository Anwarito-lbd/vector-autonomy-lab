"""HAL backend factory for seamless sim-to-real switching."""

from __future__ import annotations

from vector_lab.hal.base import VectorRobotHAL
from vector_lab.hal.simulated import SimulatedVectorRobot
from vector_lab.hal.wirepod_grpc import VectorConnectionConfig, WirePodGRPCAdapter
from vector_lab.sim.environment import Arena2D
from vector_lab.sim.kinematics import OdometryNoiseParams
from vector_lab.types import Pose2D


def create_robot_hal(
    target: str | None = None,
    *,
    host: str = "127.0.0.1",
    arena: Arena2D | None = None,
    initial_pose: Pose2D | None = None,
    initial_battery_pct: float = 82.0,
    odom_noise: OdometryNoiseParams | None = None,
    seed: int = 42,
) -> VectorRobotHAL:
    """Return SimulatedVectorRobot when target is None/'sim', else WirePodGRPCAdapter."""
    if target is None or target.strip().lower() in {"", "sim", "simulated"}:
        return SimulatedVectorRobot(
            arena=arena,
            initial_pose=initial_pose,
            initial_battery_pct=initial_battery_pct,
            odom_noise=odom_noise,
            seed=seed,
        )

    return WirePodGRPCAdapter(
        VectorConnectionConfig(
            serial=target.strip(),
            host=host,
        )
    )
