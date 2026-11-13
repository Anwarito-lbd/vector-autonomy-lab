"""Hardware Abstraction Layer (HAL) exports."""

from vector_lab.hal.base import HardwareConnectionError, VectorRobotHAL
from vector_lab.hal.factory import create_robot_hal
from vector_lab.hal.simulated import SimulatedVectorRobot
from vector_lab.hal.wirepod_grpc import VectorConnectionConfig, WirePodGRPCAdapter

__all__ = [
    "HardwareConnectionError",
    "SimulatedVectorRobot",
    "VectorConnectionConfig",
    "VectorRobotHAL",
    "WirePodGRPCAdapter",
    "create_robot_hal",
]
