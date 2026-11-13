"""Tests for log-odds Bayesian occupancy-grid SLAM and 8-connected A* planning."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from vector_lab.sim.environment import Arena2D
from vector_lab.sim.planner import AStarPlanner
from vector_lab.sim.slam import OccupancyGridConfig, OccupancyGridSLAM, bresenham_line
from vector_lab.types import Pose2D


def test_bresenham_line_connects_endpoints() -> None:
    pts = bresenham_line(2, 3, 7, 6)
    assert pts[0] == (2, 3)
    assert pts[-1] == (7, 6)
    for (x0, y0), (x1, y1) in pairwise(pts):
        assert max(abs(x1 - x0), abs(y1 - y0)) <= 1


def test_log_odds_update_clamping_and_bayes_direction() -> None:
    arena = Arena2D.default_lab_arena()
    cfg = OccupancyGridConfig(width_m=arena.width_m, height_m=arena.height_m, resolution_m=0.05)
    slam = OccupancyGridSLAM(cfg)
    pose = Pose2D(0.40, 0.30, 0.0)

    for _ in range(15):
        scan = arena.cast_range_scan(pose, timestamp_s=0.0, num_rays=72, range_noise_std_m=0.0)
        slam.update(pose, scan)

    assert float(np.min(slam.log_odds)) >= cfg.l_min - 1e-9
    assert float(np.max(slam.log_odds)) <= cfg.l_max + 1e-9

    gx, gy = slam.world_to_grid(pose.x, pose.y)
    assert slam.probabilities()[gy, gx] < cfg.p_free_thresh
    assert np.sum(slam.occupied_mask()) > 10


def test_astar_plans_collision_free_path_around_central_pillar() -> None:
    arena = Arena2D.default_lab_arena()
    slam = OccupancyGridSLAM(
        OccupancyGridConfig(width_m=arena.width_m, height_m=arena.height_m, resolution_m=0.025)
    )
    # Map arena from several vantage poses so central pillar is well observed
    vantage_poses = [
        Pose2D(0.40, 0.30, 0.0),
        Pose2D(0.45, 0.90, 0.0),
        Pose2D(1.15, 0.90, math.pi),
        Pose2D(1.15, 0.35, math.pi / 2),
        Pose2D(1.80, 0.85, math.pi),
    ]
    for vp in vantage_poses:
        scan = arena.cast_range_scan(vp, timestamp_s=0.0, num_rays=90, range_noise_std_m=0.0)
        for _ in range(3):
            slam.update(vp, scan)

    planner = AStarPlanner(inflation_radius_m=0.065)
    start = Pose2D(0.40, 0.80, 0.0)
    goal = Pose2D(1.20, 0.80, 0.0)
    path = planner.plan(slam, start, goal)

    assert len(path) >= 2
    assert path[-1].distance_to(goal) < 1e-6
    # Verify path avoids central pillar obstacle (x in [0.70, 0.92], y in [0.45, 1.18])
    for wp in path:
        assert not arena.is_occupied_ground_truth(wp.x, wp.y, margin=0.035)
