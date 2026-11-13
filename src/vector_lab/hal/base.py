"""Abstract Hardware Abstraction Layer (HAL) interface for Anki / DDL Vector."""

from __future__ import annotations

from abc import ABC, abstractmethod

from vector_lab.types import CameraFrame, RangeScan, TelemetryFrame


class HardwareConnectionError(RuntimeError):
    """Raised when a physical Vector gRPC / Wire-Pod connection cannot be established."""


class VectorRobotHAL(ABC):
    """Unified robot interface shared between SimulatedVectorRobot and Wire-Pod/gRPC hardware."""

    @abstractmethod
    def connect(self) -> None:
        """Open transport connection to the robot or simulation backend."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close transport connection and place motors into a safe idle state."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the HAL backend is currently connected."""

    @abstractmethod
    def set_wheel_speeds(self, left_mps: float, right_mps: float) -> None:
        """Command left and right tread linear velocities in m/s."""

    @abstractmethod
    def drive_twist(self, linear_mps: float, angular_rps: float) -> None:
        """Command unicycle body velocity (v m/s, omega rad/s)."""

    @abstractmethod
    def stop_motors(self) -> None:
        """Immediately zero both wheel tread velocities."""

    @abstractmethod
    def set_head_angle(self, angle_rad: float) -> None:
        """Command Vector head pitch angle in radians."""

    @abstractmethod
    def set_lift_height(self, height_norm: float) -> None:
        """Command Vector lift arm normalized height in [0.0, 1.0]."""

    @abstractmethod
    def set_eye_color(self, hue: float, saturation: float = 1.0) -> None:
        """Command Vector OLED eye hue/saturation in [0.0, 1.0]."""

    @abstractmethod
    def say_text(self, text: str) -> None:
        """Speak text via Vector TTS (or record utterance in simulation)."""

    @abstractmethod
    def read_telemetry(self) -> TelemetryFrame:
        """Return synchronized proprioceptive, battery, cliff, and proximity telemetry."""

    @abstractmethod
    def read_camera_frame(self) -> CameraFrame:
        """Capture latest grayscale/RGB camera frame and detected fiducials/faces."""

    @abstractmethod
    def read_range_scan(self) -> RangeScan:
        """Acquire latest ToF / rangefinder scan in robot body frame."""

    def step(self, dt_s: float) -> None:  # noqa: B027
        """Advance deterministic simulation clock by dt_s (no-op on real-time hardware)."""
