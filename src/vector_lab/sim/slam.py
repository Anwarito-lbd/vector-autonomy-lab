"""Bayesian log-odds 2D occupancy-grid SLAM with Bresenham raytracing and scan matching."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vector_lab.sim.environment import Arena2D
from vector_lab.types import Pose2D, RangeScan, wrap_angle


@dataclass(frozen=True, slots=True)
class OccupancyGridConfig:
    """Hyperparameters for clamped Bayesian log-odds occupancy-grid mapping."""

    width_m: float = 2.40
    height_m: float = 1.80
    resolution_m: float = 0.025
    l_occ: float = 0.85
    l_free: float = -0.35
    l_min: float = -4.0
    l_max: float = 4.0
    p_occ_thresh: float = 0.65
    p_free_thresh: float = 0.35


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Return integer grid cells along the Bresenham ray from (x0, y0) to (x1, y1)."""
    cells: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


class OccupancyGridSLAM:
    """2D log-odds occupancy grid mapper with local correlative scan matching."""

    def __init__(self, config: OccupancyGridConfig | None = None) -> None:
        self.config = config or OccupancyGridConfig()
        self.nx = math.ceil(self.config.width_m / self.config.resolution_m)
        self.ny = math.ceil(self.config.height_m / self.config.resolution_m)
        self.log_odds: NDArray[np.float64] = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.update_count: int = 0

    def in_bounds(self, gx: int, gy: int) -> bool:
        """Return True if integer cell (gx, gy) lies inside the grid."""
        return 0 <= gx < self.nx and 0 <= gy < self.ny

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates (meters) to integer cell indices (gx, gy)."""
        gx = math.floor(x / self.config.resolution_m)
        gy = math.floor(y / self.config.resolution_m)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        """Convert integer cell indices (gx, gy) to cell-center world coordinates (meters)."""
        res = self.config.resolution_m
        return ((gx + 0.5) * res, (gy + 0.5) * res)

    def update(self, pose: Pose2D, scan: RangeScan) -> None:
        """Integrate a range scan from pose into the log-odds grid using inverse sensor model."""
        gx0, gy0 = self.world_to_grid(pose.x, pose.y)
        if not self.in_bounds(gx0, gy0):
            return

        cfg = self.config
        world_angles = pose.theta + scan.angles_rad
        hit_flags = scan.valid_mask

        for angle, dist, is_hit in zip(world_angles, scan.ranges_m, hit_flags, strict=True):
            end_x = pose.x + float(dist) * math.cos(float(angle))
            end_y = pose.y + float(dist) * math.sin(float(angle))
            gx1, gy1 = self.world_to_grid(end_x, end_y)
            ray_cells = bresenham_line(gx0, gy0, gx1, gy1)
            if not ray_cells:
                continue

            free_cells = ray_cells[:-1] if is_hit else ray_cells
            for cx, cy in free_cells:
                if self.in_bounds(cx, cy):
                    self.log_odds[cy, cx] = min(
                        cfg.l_max, max(cfg.l_min, self.log_odds[cy, cx] + cfg.l_free)
                    )

            if is_hit and self.in_bounds(gx1, gy1):
                self.log_odds[gy1, gx1] = min(
                    cfg.l_max, max(cfg.l_min, self.log_odds[gy1, gx1] + cfg.l_occ)
                )

        self.update_count += 1

    def correlative_scan_match(
        self,
        pose_guess: Pose2D,
        scan: RangeScan,
        xy_window_m: float = 0.04,
        xy_step_m: float = 0.02,
        theta_window_rad: float = 0.08,
        theta_step_rad: float = 0.02,
    ) -> Pose2D:
        """Refine pose_guess against existing occupied cells via local correlative search."""
        if self.update_count < 3:
            return pose_guess

        hit_mask = scan.valid_mask
        if not np.any(hit_mask):
            return pose_guess

        occ_probs = self.probabilities()
        hit_angles = scan.angles_rad[hit_mask]
        hit_ranges = scan.ranges_m[hit_mask]

        dx_vals = np.arange(-xy_window_m, xy_window_m + 1e-9, xy_step_m)
        dy_vals = np.arange(-xy_window_m, xy_window_m + 1e-9, xy_step_m)
        dth_vals = np.arange(-theta_window_rad, theta_window_rad + 1e-9, theta_step_rad)

        res = self.config.resolution_m
        best_score = -1.0
        best_pose = pose_guess

        for dth in dth_vals:
            th = pose_guess.theta + float(dth)
            cos_a = np.cos(th + hit_angles)
            sin_a = np.sin(th + hit_angles)
            rel_x = hit_ranges * cos_a
            rel_y = hit_ranges * sin_a

            for dx in dx_vals:
                px = pose_guess.x + float(dx)
                gx = np.floor((px + rel_x) / res).astype(np.int64)
                for dy in dy_vals:
                    py = pose_guess.y + float(dy)
                    gy = np.floor((py + rel_y) / res).astype(np.int64)
                    valid = (gx >= 0) & (gx < self.nx) & (gy >= 0) & (gy < self.ny)
                    if not np.any(valid):
                        continue
                    score = float(np.sum(occ_probs[gy[valid], gx[valid]]))
                    # Mild prior favoring proximity to odometry guess
                    prior_penalty = 0.25 * (dx * dx + dy * dy) + 0.05 * (dth * dth)
                    total = score - prior_penalty
                    if total > best_score:
                        best_score = total
                        best_pose = Pose2D(x=px, y=py, theta=wrap_angle(th))

        return best_pose

    def probabilities(self) -> NDArray[np.float64]:
        """Convert log-odds grid to occupancy probabilities P(m_i) in (0, 1)."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def occupied_mask(self) -> NDArray[np.bool_]:
        """Return boolean mask of cells classified as occupied."""
        return self.probabilities() >= self.config.p_occ_thresh

    def free_mask(self) -> NDArray[np.bool_]:
        """Return boolean mask of cells classified as known free."""
        return self.probabilities() <= self.config.p_free_thresh

    def known_mask(self) -> NDArray[np.bool_]:
        """Return boolean mask of explored cells (either free or occupied)."""
        probs = self.probabilities()
        return (probs >= self.config.p_occ_thresh) | (probs <= self.config.p_free_thresh)

    def inflated_obstacles(self, inflation_radius_m: float) -> NDArray[np.bool_]:
        """Dilate occupied cells by Euclidean C-space inflation radius for safe path planning."""
        occ = self.occupied_mask()
        radius_cells = max(1, math.ceil(inflation_radius_m / self.config.resolution_m))
        inflated = occ.copy()
        occ_ys, occ_xs = np.nonzero(occ)
        if len(occ_xs) == 0:
            return inflated

        offsets: list[tuple[int, int]] = []
        r2 = radius_cells * radius_cells
        for oy in range(-radius_cells, radius_cells + 1):
            for ox in range(-radius_cells, radius_cells + 1):
                if ox * ox + oy * oy <= r2:
                    offsets.append((ox, oy))

        for ox, oy in offsets:
            ys = occ_ys + oy
            xs = occ_xs + ox
            valid = (xs >= 0) & (xs < self.nx) & (ys >= 0) & (ys < self.ny)
            inflated[ys[valid], xs[valid]] = True
        return inflated

    def evaluate_against_ground_truth(self, arena: Arena2D) -> dict[str, float]:
        """Measure explored coverage, occupied-cell precision, recall, and IoU vs ground truth."""
        res = self.config.resolution_m
        xs = (np.arange(self.nx) + 0.5) * res
        ys = (np.arange(self.ny) + 0.5) * res
        gt_occ = np.zeros((self.ny, self.nx), dtype=np.bool_)
        for gy, y in enumerate(ys):
            for gx, x in enumerate(xs):
                gt_occ[gy, gx] = arena.is_occupied_ground_truth(float(x), float(y), margin=0.0)

        pred_occ = self.occupied_mask()
        known = self.known_mask()
        coverage = float(np.mean(known))

        # Only score precision/recall/IoU over cells whose walls/obstacles were observed or marked
        tp = int(np.sum(pred_occ & gt_occ))
        fp = int(np.sum(pred_occ & ~gt_occ))
        fn = int(np.sum(~pred_occ & gt_occ & known))
        precision = tp / max(1, tp + fp)
        recall_observed = tp / max(1, tp + fn)
        iou_observed = tp / max(1, tp + fp + fn)
        return {
            "coverage_pct": round(100.0 * coverage, 2),
            "occupied_precision": round(precision, 4),
            "occupied_recall_observed": round(recall_observed, 4),
            "occupied_iou_observed": round(iou_observed, 4),
            "true_positive_cells": float(tp),
            "false_positive_cells": float(fp),
        }
