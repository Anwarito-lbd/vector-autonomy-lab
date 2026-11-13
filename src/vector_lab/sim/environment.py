"""2D/3D-aware indoor/tabletop robotics arena, ToF raycaster, cliff map, and synthetic camera."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vector_lab.types import (
    CameraFrame,
    CliffSensors,
    DetectedFace,
    DetectedMarker,
    Pose2D,
    RangeScan,
    wrap_angle,
)


@dataclass(frozen=True, slots=True)
class RectObstacle:
    """Axis-aligned 3D obstacle box projected onto the planar occupancy arena."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    height_m: float = 0.14
    label: str = "obstacle"

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Return True if point (x, y) lies inside this obstacle inflated by margin."""
        return (
            (self.x_min - margin) <= x <= (self.x_max + margin)
            and (self.y_min - margin) <= y <= (self.y_max + margin)
        )


@dataclass(frozen=True, slots=True)
class CliffZone:
    """Tabletop drop-off cutout zone where downward IR cliff sensors trigger."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str = "cliff"

    def contains(self, x: float, y: float) -> bool:
        """Return True if (x, y) is over the drop-off void."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass(frozen=True, slots=True)
class HumanAgent:
    """Simulated human face target in the arena for social greeting behavior."""

    face_id: int
    name: str
    pose: Pose2D
    height_m: float = 0.35


@dataclass(slots=True)
class Arena2D:
    """Vector tabletop/lab arena with 3D elevations, raycaster, cliff map, and synthetic camera."""

    width_m: float = 2.40
    height_m: float = 1.80
    wall_thickness_m: float = 0.04
    dock_pose: Pose2D = field(default_factory=lambda: Pose2D(0.14, 0.28, 0.0))
    dock_approach_pose: Pose2D = field(default_factory=lambda: Pose2D(0.32, 0.28, math.pi))
    obstacles: list[RectObstacle] = field(default_factory=list)
    cliffs: list[CliffZone] = field(default_factory=list)
    humans: list[HumanAgent] = field(default_factory=list)

    @classmethod
    def default_lab_arena(cls) -> Arena2D:
        """Build the canonical 2.4m x 1.8m Vector robotics research arena."""
        obstacles = [
            RectObstacle(0.70, 0.45, 0.92, 1.18, height_m=0.18, label="central_pillar"),
            RectObstacle(1.38, 0.20, 1.56, 0.72, height_m=0.14, label="south_partition"),
            RectObstacle(1.32, 1.12, 1.86, 1.32, height_m=0.16, label="north_console"),
            RectObstacle(0.45, 1.38, 0.85, 1.54, height_m=0.12, label="cube_cluster"),
        ]
        cliffs = [
            CliffZone(1.98, 0.04, 2.34, 0.34, label="table_cutout_se"),
        ]
        humans = [
            HumanAgent(
                face_id=1, name="Anwar", pose=Pose2D(1.14, 1.46, -math.pi / 2), height_m=0.38
            ),
        ]
        return cls(
            width_m=2.40,
            height_m=1.80,
            obstacles=obstacles,
            cliffs=cliffs,
            humans=humans,
        )

    def all_colliders(self) -> list[RectObstacle]:
        """Return outer bounding walls plus interior rectangular obstacles."""
        t = self.wall_thickness_m
        walls = [
            RectObstacle(0.0, 0.0, self.width_m, t, height_m=0.20, label="south_wall"),
            RectObstacle(
                0.0,
                self.height_m - t,
                self.width_m,
                self.height_m,
                height_m=0.20,
                label="north_wall",
            ),
            RectObstacle(0.0, 0.0, t, self.height_m, height_m=0.20, label="west_wall"),
            RectObstacle(
                self.width_m - t, 0.0, self.width_m, self.height_m, height_m=0.20, label="east_wall"
            ),
        ]
        return walls + self.obstacles

    def is_occupied_ground_truth(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Return True if world point (x, y) is inside a wall or obstacle."""
        if x < 0.0 or y < 0.0 or x > self.width_m or y > self.height_m:
            return True
        return any(obs.contains(x, y, margin=margin) for obs in self.all_colliders())

    def resolve_motion(
        self, prev_pose: Pose2D, candidate_pose: Pose2D, radius_m: float
    ) -> tuple[Pose2D, bool]:
        """Clamp translation on disk collision while allowing in-place rotation."""
        if self.is_occupied_ground_truth(candidate_pose.x, candidate_pose.y, margin=radius_m):
            return Pose2D(prev_pose.x, prev_pose.y, candidate_pose.theta), True
        return candidate_pose, False

    def check_cliffs(
        self, pose: Pose2D, half_length: float = 0.032, half_width: float = 0.024
    ) -> CliffSensors:
        """Evaluate Vector's 4-corner downward IR cliff detectors against arena drop-offs."""
        c = math.cos(pose.theta)
        s = math.sin(pose.theta)

        def is_drop(lx: float, ly: float) -> bool:
            wx = pose.x + lx * c - ly * s
            wy = pose.y + lx * s + ly * c
            margin = self.wall_thickness_m * 0.5
            if wx <= margin or wy <= margin:
                return True
            if wx >= self.width_m - margin or wy >= self.height_m - margin:
                return True
            return any(cliff.contains(wx, wy) for cliff in self.cliffs)

        return CliffSensors(
            front_left=is_drop(half_length, half_width),
            front_right=is_drop(half_length, -half_width),
            back_left=is_drop(-half_length, half_width),
            back_right=is_drop(-half_length, -half_width),
        )

    def cast_range_scan(
        self,
        pose: Pose2D,
        timestamp_s: float,
        num_rays: int = 72,
        fov_rad: float = 2.0 * math.pi,
        max_range_m: float = 2.0,
        range_noise_std_m: float = 0.005,
        rng: np.random.Generator | None = None,
    ) -> RangeScan:
        """Cast vectorized 2D/3D-projected ToF laser rays against arena geometry."""
        if fov_rad >= 2.0 * math.pi - 1e-6:
            angles = np.linspace(-math.pi, math.pi, num_rays, endpoint=False, dtype=np.float64)
        else:
            angles = np.linspace(-0.5 * fov_rad, 0.5 * fov_rad, num_rays, dtype=np.float64)

        world_angles = pose.theta + angles
        dx = np.cos(world_angles)
        dy = np.sin(world_angles)

        # Slab ray-AABB intersection across all colliders simultaneously
        colliders = self.all_colliders()
        x_mins = np.array([o.x_min for o in colliders], dtype=np.float64)[:, None]
        x_maxs = np.array([o.x_max for o in colliders], dtype=np.float64)[:, None]
        y_mins = np.array([o.y_min for o in colliders], dtype=np.float64)[:, None]
        y_maxs = np.array([o.y_max for o in colliders], dtype=np.float64)[:, None]

        inv_dx = np.where(
            np.abs(dx) > 1e-12, 1.0 / np.where(np.abs(dx) > 1e-12, dx, 1.0), 0.0
        )[None, :]
        inv_dy = np.where(
            np.abs(dy) > 1e-12, 1.0 / np.where(np.abs(dy) > 1e-12, dy, 1.0), 0.0
        )[None, :]

        t1x = (x_mins - pose.x) * inv_dx
        t2x = (x_maxs - pose.x) * inv_dx
        t1y = (y_mins - pose.y) * inv_dy
        t2y = (y_maxs - pose.y) * inv_dy

        # Handle rays exactly parallel to axes cleanly
        parallel_x = np.abs(dx) <= 1e-12
        parallel_y = np.abs(dy) <= 1e-12
        inside_x = (pose.x >= x_mins) & (pose.x <= x_maxs)
        inside_y = (pose.y >= y_mins) & (pose.y <= y_maxs)

        tmin_x = np.where(
            parallel_x[None, :], np.where(inside_x, -np.inf, np.inf), np.minimum(t1x, t2x)
        )
        tmax_x = np.where(
            parallel_x[None, :], np.where(inside_x, np.inf, -np.inf), np.maximum(t1x, t2x)
        )
        tmin_y = np.where(
            parallel_y[None, :], np.where(inside_y, -np.inf, np.inf), np.minimum(t1y, t2y)
        )
        tmax_y = np.where(
            parallel_y[None, :], np.where(inside_y, np.inf, -np.inf), np.maximum(t1y, t2y)
        )

        t_enter = np.maximum(tmin_x, tmin_y)
        t_exit = np.minimum(tmax_x, tmax_y)

        valid_hit = (
            (t_exit >= np.maximum(t_enter, 0.0)) & (t_enter >= 0.0) & (t_enter <= max_range_m)
        )
        hit_dists = np.where(valid_hit, t_enter, max_range_m)
        ranges = np.min(hit_dists, axis=0)

        if rng is not None and range_noise_std_m > 0.0:
            hit_mask = ranges < (max_range_m - 1e-4)
            noise = rng.normal(0.0, range_noise_std_m, size=ranges.shape)
            ranges = np.where(hit_mask, np.clip(ranges + noise, 0.02, max_range_m), max_range_m)

        elevations = np.zeros_like(angles)
        return RangeScan(
            timestamp_s=timestamp_s,
            angles_rad=angles,
            ranges_m=ranges,
            max_range_m=max_range_m,
            elevations_rad=elevations,
        )

    def _has_line_of_sight(self, p0: Pose2D, p1: Pose2D, step_m: float = 0.03) -> bool:
        """Return True if the segment from p0 to p1 is free of opaque obstacles."""
        dist = p0.distance_to(p1)
        if dist < 1e-6:
            return True
        steps = max(2, int(dist / step_m))
        for i in range(1, steps):
            u = i / steps
            x = p0.x + u * (p1.x - p0.x)
            y = p0.y + u * (p1.y - p0.y)
            if self.is_occupied_ground_truth(x, y, margin=0.0):
                return False
        return True

    def render_synthetic_camera(
        self,
        pose: Pose2D,
        timestamp_s: float,
        width: int = 160,
        height: int = 120,
        hfov_rad: float = math.radians(90.0),
    ) -> CameraFrame:
        """Render a deterministic synthetic grayscale view with faces & dock fiducials."""
        img = np.full((height, width), 175, dtype=np.uint8)
        # Floor gradient in lower half of image
        floor_rows = np.linspace(120, 85, height // 2, dtype=np.uint8)[:, None]
        img[height // 2 :, :] = floor_rows

        # Cast column rays across camera horizontal FOV to shade walls/obstacles by depth
        col_angles = np.linspace(0.5 * hfov_rad, -0.5 * hfov_rad, width, dtype=np.float64)
        scan = self.cast_range_scan(
            pose=pose,
            timestamp_s=timestamp_s,
            num_rays=width,
            fov_rad=hfov_rad,
            max_range_m=2.5,
            range_noise_std_m=0.0,
        )
        # Re-order because scan angles go -fov/2 .. +fov/2 while image x=0 is left (+fov/2)
        col_ranges = scan.ranges_m[::-1]
        for x in range(width):
            dist = max(0.08, float(col_ranges[x]) * math.cos(col_angles[x]))
            wall_h = int(min(height - 4, max(8, (0.24 / dist) * (height * 0.55))))
            y0 = max(0, height // 2 - wall_h // 2)
            y1 = min(height, height // 2 + wall_h // 2)
            shade = int(np.clip(210.0 - 75.0 * min(dist, 2.2), 45, 200))
            img[y0:y1, x] = shade

        faces: list[DetectedFace] = []
        for human in self.humans:
            dist = pose.distance_to(human.pose)
            bearing = wrap_angle(pose.bearing_to(human.pose) - pose.theta)
            if (
                dist <= 1.15
                and abs(bearing) <= 0.45 * hfov_rad
                and self._has_line_of_sight(pose, human.pose)
            ):
                cx = int((0.5 - bearing / hfov_rad) * width)
                box_w = int(np.clip(28.0 / max(dist, 0.2), 14, 52))
                box_h = int(box_w * 1.2)
                x0 = int(np.clip(cx - box_w // 2, 0, width - box_w))
                y0 = int(np.clip(height // 2 - box_h // 2 - 8, 0, height - box_h))
                # Draw high-contrast synthetic face silhouette + eye contrast for CV detector
                img[y0 : y0 + box_h, x0 : x0 + box_w] = 228
                eye_y = y0 + box_h // 3
                img[eye_y : eye_y + 3, x0 + box_w // 4 : x0 + box_w // 4 + 3] = 30
                img[eye_y : eye_y + 3, x0 + (3 * box_w) // 4 - 3 : x0 + (3 * box_w) // 4] = 30
                faces.append(
                    DetectedFace(
                        face_id=human.face_id,
                        name=human.name,
                        bbox_xywh=(x0, y0, box_w, box_h),
                        distance_m=dist,
                        bearing_rad=bearing,
                        confidence=float(np.clip(0.98 - 0.15 * dist, 0.65, 0.99)),
                    )
                )

        markers: list[DetectedMarker] = []
        dock_dist = pose.distance_to(self.dock_pose)
        dock_bearing = wrap_angle(pose.bearing_to(self.dock_pose) - pose.theta)
        if (
            dock_dist <= 1.25
            and abs(dock_bearing) <= 0.45 * hfov_rad
            and self._has_line_of_sight(pose, self.dock_pose)
        ):
            cx = int((0.5 - dock_bearing / hfov_rad) * width)
            bw = int(np.clip(22.0 / max(dock_dist, 0.2), 10, 42))
            x0 = int(np.clip(cx - bw // 2, 0, width - bw))
            y0 = int(np.clip(height // 2 - bw // 2 + 4, 0, height - bw))
            # Draw high-contrast AprilTag-like checkerboard fiducial
            img[y0 : y0 + bw, x0 : x0 + bw] = 25
            half = max(2, bw // 2)
            img[y0 : y0 + half, x0 : x0 + half] = 240
            img[y0 + half : y0 + bw, x0 + half : x0 + bw] = 240
            markers.append(
                DetectedMarker(
                    marker_id="CHARGER_DOCK_TAG",
                    kind="charger_dock",
                    bbox_xywh=(x0, y0, bw, bw),
                    distance_m=dock_dist,
                    bearing_rad=dock_bearing,
                    confidence=float(np.clip(0.99 - 0.12 * dock_dist, 0.70, 0.99)),
                )
            )

        return CameraFrame(
            timestamp_s=timestamp_s,
            width=width,
            height=height,
            image=img,
            brightness=float(np.mean(img) / 255.0),
            faces=tuple(faces),
            markers=tuple(markers),
        )
