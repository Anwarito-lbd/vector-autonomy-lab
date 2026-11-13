"""Core robotics, sensor, perception, and behavior-tree blackboard data structures."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from vector_lab.sim.slam import OccupancyGridSLAM


def wrap_angle(theta: float) -> float:
    """Wrap an angle in radians to the principal interval (-pi, pi]."""
    wrapped = (theta + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == -math.pi:
        return math.pi
    return wrapped


@dataclass(frozen=True, slots=True)
class Pose2D:
    """Planar SE(2) robot pose in meters and radians."""

    x: float
    y: float
    theta: float

    def distance_to(self, other: Pose2D) -> float:
        """Return Euclidean planar distance to another pose in meters."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def bearing_to(self, other: Pose2D) -> float:
        """Return world-frame angle from this pose toward another pose."""
        return math.atan2(other.y - self.y, other.x - self.x)

    def heading_error_to(self, other: Pose2D) -> float:
        """Return signed angular error wrapped to (-pi, pi] toward target pose."""
        return wrap_angle(self.bearing_to(other) - self.theta)

    def as_tuple(self) -> tuple[float, float, float]:
        """Return (x, y, theta) tuple."""
        return (self.x, self.y, self.theta)


@dataclass(frozen=True, slots=True)
class Twist2D:
    """Planar unicycle body velocity command (linear m/s, angular rad/s)."""

    v: float
    omega: float


@dataclass(frozen=True, slots=True)
class WheelVelocities:
    """Differential-drive left/right tread linear speeds in m/s."""

    v_left: float
    v_right: float


@dataclass(frozen=True, slots=True)
class CliffSensors:
    """Four-corner infrared cliff detector states on Vector's chassis."""

    front_left: bool = False
    front_right: bool = False
    back_left: bool = False
    back_right: bool = False

    @property
    def any_triggered(self) -> bool:
        """Return True if any cliff sensor detects a drop-off."""
        return self.front_left or self.front_right or self.back_left or self.back_right

    @property
    def front_triggered(self) -> bool:
        """Return True if either front cliff sensor triggers."""
        return self.front_left or self.front_right


@dataclass(frozen=True, slots=True)
class TelemetryFrame:
    """Synchronized robot proprioceptive and safety telemetry snapshot."""

    timestamp_s: float
    pose_odom: Pose2D
    wheel_speeds: WheelVelocities
    battery_voltage: float
    battery_pct: float
    is_charging: bool
    on_charger: bool
    cliff: CliffSensors
    proximity_mm: float
    head_angle_rad: float
    lift_height_norm: float
    imu_accel_mps2: tuple[float, float, float] = (0.0, 0.0, 9.81)
    imu_gyro_rps: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """Detected human face in camera image space with estimated metric range."""

    face_id: int
    name: str
    bbox_xywh: tuple[int, int, int, int]
    distance_m: float
    bearing_rad: float
    confidence: float


@dataclass(frozen=True, slots=True)
class DetectedMarker:
    """Detected fiducial marker (e.g., charger AprilTag or LightCube beacon)."""

    marker_id: str
    kind: str
    bbox_xywh: tuple[int, int, int, int]
    distance_m: float
    bearing_rad: float
    confidence: float


@dataclass(slots=True)
class CameraFrame:
    """Grayscale/RGB camera frame with optional synthetic ground-truth annotations."""

    timestamp_s: float
    width: int
    height: int
    image: NDArray[np.uint8]
    brightness: float
    faces: tuple[DetectedFace, ...] = ()
    markers: tuple[DetectedMarker, ...] = ()


@dataclass(slots=True)
class RangeScan:
    """Planar 2D/3D-projected ToF / LiDAR range scan in robot body frame."""

    timestamp_s: float
    angles_rad: NDArray[np.float64]
    ranges_m: NDArray[np.float64]
    max_range_m: float
    elevations_rad: NDArray[np.float64] | None = None

    @property
    def valid_mask(self) -> NDArray[np.bool_]:
        """Return mask of rays that recorded a physical obstacle hit (< max_range)."""
        return self.ranges_m < (self.max_range_m - 1e-4)

    @property
    def min_forward_range_m(self) -> float:
        """Return minimum range within the forward +-30 deg cone."""
        forward_mask = np.abs(self.angles_rad) <= math.radians(30.0)
        if not np.any(forward_mask):
            return float(self.max_range_m)
        return float(np.min(self.ranges_m[forward_mask]))


class IntentType(StrEnum):
    """Structured voice/operator autonomy intent classes."""

    PATROL = "PATROL"
    GOTO_POSE = "GOTO_POSE"
    RETURN_TO_DOCK = "RETURN_TO_DOCK"
    GREET_PERSON = "GREET_PERSON"
    STOP = "STOP"
    STATUS_REPORT = "STATUS_REPORT"


@dataclass(frozen=True, slots=True)
class IntentGoal:
    """Parsed high-level intent ready for Behavior Tree blackboard dispatch."""

    intent: IntentType
    utterance: str
    target_pose: Pose2D | None = None
    person_name: str | None = None
    confidence: float = 1.0
    source: str = "deterministic"


@dataclass(slots=True)
class Blackboard:
    """Shared deterministic state bus connecting HAL, Perception, SLAM, and Behavior Tree."""

    tick_count: int = 0
    sim_time_s: float = 0.0
    telemetry: TelemetryFrame | None = None
    camera: CameraFrame | None = None
    scan: RangeScan | None = None
    slam_map: OccupancyGridSLAM | None = None
    estimated_pose: Pose2D = field(default_factory=lambda: Pose2D(0.0, 0.0, 0.0))

    # Navigation & Patrol state
    patrol_waypoints: list[Pose2D] = field(default_factory=list)
    patrol_index: int = 0
    dock_approach_pose: Pose2D = field(default_factory=lambda: Pose2D(0.25, 0.25, math.pi))
    dock_pose: Pose2D = field(default_factory=lambda: Pose2D(0.14, 0.25, math.pi))
    active_goal: Pose2D | None = None
    planned_path: list[Pose2D] = field(default_factory=list)
    path_cursor: int = 0

    # Perception & Intent state
    pending_intent: IntentGoal | None = None
    active_intent: IntentGoal | None = None
    detected_faces: list[DetectedFace] = field(default_factory=list)
    detected_markers: list[DetectedMarker] = field(default_factory=list)
    forward_obstacle_dist_m: float = 2.0
    visual_edge_density: float = 0.0

    # Safety & Social timers
    low_battery_threshold_pct: float = 25.0
    charged_threshold_pct: float = 90.0
    safe_obstacle_dist_m: float = 0.14
    cliff_recovery_ticks: int = 0
    face_greet_cooldown_until_s: float = 0.0
    halted_by_operator: bool = False

    # Observable execution traces
    active_branch: str = "IDLE"
    last_spoken: str = ""
    eye_color_hue: float = 0.45
    event_log: list[tuple[float, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def log_event(self, message: str) -> None:
        """Append a timestamped event to the blackboard audit log."""
        self.event_log.append((round(self.sim_time_s, 3), message))
