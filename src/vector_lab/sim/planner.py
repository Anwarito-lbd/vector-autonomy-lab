"""8-connected grid A* path planner with C-space inflation and line-of-sight pruning."""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np
from numpy.typing import NDArray

from vector_lab.sim.slam import OccupancyGridSLAM, bresenham_line
from vector_lab.types import Pose2D

_NEIGHBORS_8: tuple[tuple[int, int, float], ...] = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (-1, -1, math.sqrt(2.0)),
)


class AStarPlanner:
    """Grid-based A* planner operating over SLAM occupancy grids with string-pulling."""

    def __init__(self, inflation_radius_m: float = 0.065) -> None:
        self.inflation_radius_m = inflation_radius_m

    @staticmethod
    def _octile_heuristic(ax: int, ay: int, bx: int, by: int) -> float:
        dx = abs(ax - bx)
        dy = abs(ay - by)
        d_min = min(dx, dy)
        d_max = max(dx, dy)
        return math.sqrt(2.0) * d_min + (d_max - d_min)

    @staticmethod
    def _nearest_free_cell(
        start_cell: tuple[int, int],
        blocked: NDArray[np.bool_],
        max_radius: int = 8,
    ) -> tuple[int, int] | None:
        ny, nx = blocked.shape
        sx = int(np.clip(start_cell[0], 0, nx - 1))
        sy = int(np.clip(start_cell[1], 0, ny - 1))
        if not blocked[sy, sx]:
            return (sx, sy)

        visited = {(sx, sy)}
        queue: deque[tuple[int, int, int]] = deque([(sx, sy, 0)])
        while queue:
            cx, cy, dist = queue.popleft()
            if dist >= max_radius:
                continue
            for dx, dy, _ in _NEIGHBORS_8:
                nx_c, ny_c = cx + dx, cy + dy
                if 0 <= nx_c < nx and 0 <= ny_c < ny and (nx_c, ny_c) not in visited:
                    if not blocked[ny_c, nx_c]:
                        return (nx_c, ny_c)
                    visited.add((nx_c, ny_c))
                    queue.append((nx_c, ny_c, dist + 1))
        return None

    @staticmethod
    def _prune_path(
        cells: list[tuple[int, int]],
        blocked: NDArray[np.bool_],
    ) -> list[tuple[int, int]]:
        """Remove redundant waypoints when direct Bresenham sight is obstacle-free."""
        if len(cells) <= 2:
            return cells
        pruned: list[tuple[int, int]] = [cells[0]]
        anchor_idx = 0
        while anchor_idx < len(cells) - 1:
            farthest = anchor_idx + 1
            for probe in range(len(cells) - 1, anchor_idx + 1, -1):
                line = bresenham_line(
                    cells[anchor_idx][0],
                    cells[anchor_idx][1],
                    cells[probe][0],
                    cells[probe][1],
                )
                if all(not blocked[cy, cx] for cx, cy in line):
                    farthest = probe
                    break
            pruned.append(cells[farthest])
            anchor_idx = farthest
        return pruned

    def plan(
        self,
        slam: OccupancyGridSLAM,
        start: Pose2D,
        goal: Pose2D,
        prune: bool = True,
    ) -> list[Pose2D]:
        """Compute collision-free Pose2D waypoints from start to goal on the live SLAM grid."""
        blocked = slam.inflated_obstacles(self.inflation_radius_m)
        start_cell = self._nearest_free_cell(slam.world_to_grid(start.x, start.y), blocked)
        goal_cell = self._nearest_free_cell(slam.world_to_grid(goal.x, goal.y), blocked)
        if start_cell is None or goal_cell is None:
            return []
        if start_cell == goal_cell:
            return [goal]

        ny, nx = blocked.shape
        g_score: dict[tuple[int, int], float] = {start_cell: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        h_cost = self._octile_heuristic(
            start_cell[0], start_cell[1], goal_cell[0], goal_cell[1]
        )
        heap: list[tuple[float, int, tuple[int, int]]] = [(h_cost, 0, start_cell)]
        counter = 1
        closed: set[tuple[int, int]] = set()

        while heap:
            _, _, current = heapq.heappop(heap)
            if current in closed:
                continue
            if current == goal_cell:
                raw_cells: list[tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    raw_cells.append(current)
                raw_cells.reverse()
                final_cells = self._prune_path(raw_cells, blocked) if prune else raw_cells
                return self._cells_to_poses(slam, final_cells, goal)

            closed.add(current)
            cx, cy = current
            cur_g = g_score[current]

            for dx, dy, step_cost in _NEIGHBORS_8:
                nx_c, ny_c = cx + dx, cy + dy
                if not (0 <= nx_c < nx and 0 <= ny_c < ny):
                    continue
                if blocked[ny_c, nx_c]:
                    continue
                # Prevent diagonal corner-cutting across occupied orthogonal neighbors
                if dx != 0 and dy != 0 and (blocked[cy, nx_c] or blocked[ny_c, cx]):
                    continue

                nxt = (nx_c, ny_c)
                tentative_g = cur_g + step_cost
                if tentative_g < g_score.get(nxt, math.inf):
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    f_score = tentative_g + self._octile_heuristic(
                        nx_c, ny_c, goal_cell[0], goal_cell[1]
                    )
                    heapq.heappush(heap, (f_score, counter, nxt))
                    counter += 1

        return []

    @staticmethod
    def _cells_to_poses(
        slam: OccupancyGridSLAM,
        cells: list[tuple[int, int]],
        final_goal: Pose2D,
    ) -> list[Pose2D]:
        if not cells:
            return []
        world_pts = [slam.grid_to_world(gx, gy) for gx, gy in cells[1:]]
        poses: list[Pose2D] = []
        for idx, (wx, wy) in enumerate(world_pts):
            if idx == len(world_pts) - 1:
                poses.append(Pose2D(x=final_goal.x, y=final_goal.y, theta=final_goal.theta))
            else:
                nx, ny = world_pts[idx + 1]
                poses.append(Pose2D(x=wx, y=wy, theta=math.atan2(ny - wy, nx - wx)))
        return poses or [final_goal]
