"""Multi-episode deterministic simulations generating SLAM, kinematics, BT plots & GIF."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from vector_lab.sim.environment import Arena2D
from vector_lab.sim.kinematics import OdometryNoiseParams
from vector_lab.sim.planner import AStarPlanner
from vector_lab.sim.runner import EpisodeResult, SimulationEpisodeRunner
from vector_lab.sim.slam import OccupancyGridConfig, OccupancyGridSLAM
from vector_lab.types import Pose2D

matplotlib.use("Agg")

_BRANCH_ORDER = [
    "PATROL",
    "FACE_GREET",
    "VOICE_GOTO",
    "VOICE_STATUS",
    "OBSTACLE_AVOID",
    "CLIFF_AVOID",
    "DOCK_CHARGE",
    "OPERATOR_STOP",
]

_BRANCH_COLORS = {
    "PATROL": "#00bcd4",
    "FACE_GREET": "#4caf50",
    "VOICE_GOTO": "#7c4dff",
    "VOICE_STATUS": "#26a69a",
    "OBSTACLE_AVOID": "#ff9800",
    "CLIFF_AVOID": "#f44336",
    "DOCK_CHARGE": "#ffca28",
    "OPERATOR_STOP": "#90a4ae",
}


def _configure_dark_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0f172a",
            "axes.facecolor": "#111827",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#e2e8f0",
            "xtick.color": "#94a3b8",
            "ytick.color": "#94a3b8",
            "text.color": "#f8fafc",
            "grid.color": "#1e293b",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )


def plot_slam_occupancy_trajectory(ep: EpisodeResult, out_path: Path) -> None:
    """Plot reconstructed log-odds occupancy probability grid, ground-truth geometry & paths."""
    _configure_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.0), dpi=180)
    arena = ep.arena
    probs = ep.slam.probabilities()
    extent = [0.0, arena.width_m, 0.0, arena.height_m]

    ax0 = axes[0]
    ax0.set_title(
        "Arena Geometry, Ground Truth vs. Odometry & SLAM",
        pad=10,
        fontsize=11,
        fontweight="bold",
    )
    for obs in arena.all_colliders():
        rect = patches.Rectangle(
            (obs.x_min, obs.y_min),
            obs.x_max - obs.x_min,
            obs.y_max - obs.y_min,
            facecolor="#334155" if "wall" in obs.label else "#475569",
            edgecolor="#94a3b8",
            linewidth=0.8,
        )
        ax0.add_patch(rect)

    for cliff in arena.cliffs:
        crect = patches.Rectangle(
            (cliff.x_min, cliff.y_min),
            cliff.x_max - cliff.x_min,
            cliff.y_max - cliff.y_min,
            facecolor="#7f1d1d",
            edgecolor="#ef4444",
            hatch="///",
            alpha=0.65,
            label="Table Cliff Drop-Off",
        )
        ax0.add_patch(crect)

    gt_x = [r.gt_pose.x for r in ep.records]
    gt_y = [r.gt_pose.y for r in ep.records]
    od_x = [r.odom_pose.x for r in ep.records]
    od_y = [r.odom_pose.y for r in ep.records]
    sl_x = [r.slam_pose.x for r in ep.records]
    sl_y = [r.slam_pose.y for r in ep.records]

    ax0.plot(
        od_x,
        od_y,
        color="#f97316",
        linestyle="--",
        linewidth=1.4,
        label="Raw Odometry (Drift)",
    )
    ax0.plot(sl_x, sl_y, color="#38bdf8", linewidth=1.8, alpha=0.9, label="SLAM Scan-Matched")
    ax0.plot(gt_x, gt_y, color="#22c55e", linewidth=1.6, alpha=0.85, label="Ground Truth")

    ax0.scatter(
        [arena.dock_pose.x],
        [arena.dock_pose.y],
        marker="s",
        s=120,
        color="#facc15",
        edgecolor="#0f172a",
        zorder=6,
        label="Charging Dock",
    )
    for h in arena.humans:
        ax0.scatter(
            [h.pose.x],
            [h.pose.y],
            marker="*",
            s=170,
            color="#e879f9",
            edgecolor="#0f172a",
            zorder=6,
            label=f"Human ({h.name})",
        )

    ax0.set_xlim(0.0, arena.width_m)
    ax0.set_ylim(0.0, arena.height_m)
    ax0.set_aspect("equal")
    ax0.set_xlabel("X [m]")
    ax0.set_ylabel("Y [m]")
    ax0.grid(True)
    ax0.legend(loc="upper right", fontsize=8, framealpha=0.88)

    ax1 = axes[1]
    cov = ep.slam_metrics["coverage_pct"]
    prec = ep.slam_metrics["occupied_precision"]
    ax1.set_title(
        f"Occupancy Grid SLAM (Cov. {cov:.1f}%, Prec. {prec:.3f})",
        pad=10,
        fontsize=11,
        fontweight="bold",
    )
    im = ax1.imshow(
        probs,
        origin="lower",
        extent=extent,
        cmap="bone_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.042, pad=0.03)
    cbar.set_label("Occupancy Probability P(m_i)")

    planner = AStarPlanner(inflation_radius_m=0.065)
    sample_plan = planner.plan(ep.slam, Pose2D(1.96, 1.44, math.pi), arena.dock_approach_pose)
    if sample_plan:
        px = [1.96] + [p.x for p in sample_plan]
        py = [1.44] + [p.y for p in sample_plan]
        ax1.plot(
            px,
            py,
            color="#f43f5e",
            linewidth=2.2,
            linestyle="-.",
            label="A* Pruned Path -> Dock",
        )
        ax1.scatter(px, py, color="#fda4af", s=18, zorder=5)

    ax1.plot(sl_x, sl_y, color="#38bdf8", linewidth=1.3, alpha=0.85, label="SLAM Trajectory")
    ax1.set_xlim(0.0, arena.width_m)
    ax1.set_ylim(0.0, arena.height_m)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.88)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_kinematics_odometry_error(
    ideal_ep: EpisodeResult,
    nominal_no_slam: EpisodeResult,
    nominal_slam: EpisodeResult,
    high_slip_no_slam: EpisodeResult,
    high_slip_slam: EpisodeResult,
    out_path: Path,
) -> None:
    """Plot differential-drive localization & heading errors across slip regimes."""
    _configure_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=180)

    ax0 = axes[0]
    t_nom = [r.time_s for r in nominal_no_slam.records]
    ax0.plot(
        t_nom,
        [r.odom_pos_err_m * 100.0 for r in nominal_no_slam.records],
        color="#f97316",
        linewidth=1.7,
        label="Nominal Raw Odom",
    )
    ax0.plot(
        [r.time_s for r in nominal_slam.records],
        [r.slam_pos_err_m * 100.0 for r in nominal_slam.records],
        color="#38bdf8",
        linewidth=1.8,
        label="Nominal + SLAM ScanMatch",
    )
    ax0.plot(
        [r.time_s for r in high_slip_no_slam.records],
        [r.odom_pos_err_m * 100.0 for r in high_slip_no_slam.records],
        color="#ef4444",
        linestyle="--",
        linewidth=1.5,
        label="High-Slip Raw Odom",
    )
    ax0.plot(
        [r.time_s for r in high_slip_slam.records],
        [r.slam_pos_err_m * 100.0 for r in high_slip_slam.records],
        color="#22c55e",
        linewidth=1.7,
        label="High-Slip + SLAM ScanMatch",
    )
    ax0.set_title("Planar Localization Error vs. Time", fontweight="bold")
    ax0.set_xlabel("Simulation Time [s]")
    ax0.set_ylabel("Position Error [cm]")
    ax0.grid(True)
    ax0.legend(fontsize=8)

    ax1 = axes[1]
    ax1.plot(
        t_nom,
        [math.degrees(r.odom_heading_err_rad) for r in nominal_no_slam.records],
        color="#f97316",
        linewidth=1.7,
        label="Nominal Raw Odom",
    )
    ax1.plot(
        [r.time_s for r in nominal_slam.records],
        [math.degrees(r.slam_heading_err_rad) for r in nominal_slam.records],
        color="#38bdf8",
        linewidth=1.8,
        label="Nominal + SLAM ScanMatch",
    )
    ax1.plot(
        [r.time_s for r in high_slip_no_slam.records],
        [math.degrees(r.odom_heading_err_rad) for r in high_slip_no_slam.records],
        color="#ef4444",
        linestyle="--",
        linewidth=1.5,
        label="High-Slip Raw Odom",
    )
    ax1.plot(
        [r.time_s for r in high_slip_slam.records],
        [math.degrees(r.slam_heading_err_rad) for r in high_slip_slam.records],
        color="#22c55e",
        linewidth=1.7,
        label="High-Slip + SLAM ScanMatch",
    )
    ax1.set_title("Heading Drift Error vs. Time", fontweight="bold")
    ax1.set_xlabel("Simulation Time [s]")
    ax1.set_ylabel("Absolute Heading Error [deg]")
    ax1.grid(True)
    ax1.legend(fontsize=8)

    ax2 = axes[2]
    labels = [
        "Ideal\n(No Slip)",
        "Nominal\nOdom",
        "Nominal\n+ SLAM",
        "High-Slip\nOdom",
        "High-Slip\n+ SLAM",
    ]
    means_cm = [
        ideal_ep.mean_odom_error_m * 100.0,
        nominal_no_slam.mean_odom_error_m * 100.0,
        nominal_slam.mean_slam_error_m * 100.0,
        high_slip_no_slam.mean_odom_error_m * 100.0,
        high_slip_slam.mean_slam_error_m * 100.0,
    ]
    colors = ["#94a3b8", "#f97316", "#38bdf8", "#ef4444", "#22c55e"]
    bars = ax2.bar(labels, means_cm, color=colors, edgecolor="#0f172a", width=0.62)
    for bar, val in zip(bars, means_cm, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.35,
            f"{val:.2f} cm",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.set_title("Mean Error Across Slip Regimes", fontweight="bold")
    ax2.set_ylabel("Mean Position Error [cm]")
    ax2.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_behavior_tree_timeline(
    main_ep: EpisodeResult,
    cliff_ep: EpisodeResult,
    out_path: Path,
) -> None:
    """Plot active BT branch preemption, battery & proximity telemetry, and cliff recovery."""
    _configure_dark_style()
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 8.8), dpi=180, sharex=False)

    times = [r.time_s for r in main_ep.records]
    branch_idx_map = {name: i for i, name in enumerate(_BRANCH_ORDER)}

    ax0 = axes[0]
    y_vals = [branch_idx_map.get(r.active_branch, 0) for r in main_ep.records]
    pt_colors = [_BRANCH_COLORS.get(r.active_branch, "#94a3b8") for r in main_ep.records]
    ax0.step(times, y_vals, where="post", color="#475569", linewidth=1.0, alpha=0.6)
    ax0.scatter(times, y_vals, c=pt_colors, s=14, zorder=4)
    ax0.set_yticks(range(len(_BRANCH_ORDER)))
    ax0.set_yticklabels(_BRANCH_ORDER, fontsize=8.5)
    ax0.set_title(
        "Episode 1: Behavior-Tree Reactive Preemption Timeline (Patrol -> Voice -> Face -> Dock)",
        fontweight="bold",
    )
    ax0.set_ylabel("Active BT Branch")
    ax0.grid(True)

    ax1 = axes[1]
    bat = [r.battery_pct for r in main_ep.records]
    prox = [r.min_forward_range_m for r in main_ep.records]
    ax1.plot(times, bat, color="#facc15", linewidth=2.0, label="Battery [%]")
    ax1.axhline(
        25.0,
        color="#ef4444",
        linestyle="--",
        linewidth=1.1,
        label="Low-Battery Threshold (25%)",
    )
    ax1.set_ylabel("Battery [%]", color="#facc15")
    ax1.set_ylim(0, 105)
    ax1.grid(True)

    ax1_r = ax1.twinx()
    ax1_r.plot(times, prox, color="#38bdf8", linewidth=1.4, alpha=0.85, label="Min Forward ToF [m]")
    ax1_r.axhline(0.14, color="#fb923c", linestyle=":", linewidth=1.1, label="Safety Dist (0.14m)")
    ax1_r.set_ylabel("Forward ToF Range [m]", color="#38bdf8")
    ax1_r.tick_params(axis="y", colors="#38bdf8")
    ax1.set_title("Energy State & Forward Rangefinder Preemption Triggers", fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax2 = axes[2]
    t2 = [r.time_s for r in cliff_ep.records]
    v2 = [r.linear_v_mps for r in cliff_ep.records]
    w2 = [r.angular_omega_rps for r in cliff_ep.records]
    ax2.plot(t2, v2, color="#22c55e", linewidth=1.7, label="Linear Velocity v [m/s]")
    ax2.plot(
        t2,
        [w * 0.08 for w in w2],
        color="#a855f7",
        linewidth=1.3,
        alpha=0.85,
        label="0.08 * Angular Rate [rad/s]",
    )
    cliff_times = [r.time_s for r in cliff_ep.records if r.active_branch == "CLIFF_AVOID"]
    for ct in cliff_times:
        ax2.axvspan(ct - 0.05, ct + 0.05, color="#ef4444", alpha=0.22)
    if cliff_times:
        ax2.axvline(
            cliff_times[0],
            color="#ef4444",
            linestyle="--",
            linewidth=1.2,
            label="Cliff Preemption Window",
        )
    ax2.set_title(
        "Episode 2: Table-Cutout Cliff Detection & Reverse Recovery Control",
        fontweight="bold",
    )
    ax2.set_xlabel("Simulation Time [s]")
    ax2.set_ylabel("Control Effort")
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_slam_exploration_gif(ep: EpisodeResult, out_path: Path) -> None:
    """Render an animated GIF showing live log-odds SLAM discovery, ToF rays & trajectory."""
    _configure_dark_style()
    arena = ep.arena
    slam_replay = OccupancyGridSLAM(
        OccupancyGridConfig(
            width_m=arena.width_m,
            height_m=arena.height_m,
            resolution_m=0.025,
        )
    )

    # Precompute progressive frames at stride=4 (~160 frames)
    stride = 4
    frames_data: list[tuple[np.ndarray, int, float]] = []
    for idx, rec in enumerate(ep.records):
        scan = arena.cast_range_scan(
            pose=rec.gt_pose,
            timestamp_s=rec.time_s,
            num_rays=72,
            fov_rad=2.0 * math.pi,
            max_range_m=2.0,
            range_noise_std_m=0.0,
        )
        slam_replay.update(rec.slam_pose, scan)
        if idx % stride == 0 or idx == len(ep.records) - 1:
            cov = float(np.mean(slam_replay.known_mask())) * 100.0
            frames_data.append((slam_replay.probabilities().copy(), idx, cov))

    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=115)
    extent = [0.0, arena.width_m, 0.0, arena.height_m]
    im = ax.imshow(
        frames_data[0][0],
        origin="lower",
        extent=extent,
        cmap="bone_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    for cliff in arena.cliffs:
        ax.add_patch(
            patches.Rectangle(
                (cliff.x_min, cliff.y_min),
                cliff.x_max - cliff.x_min,
                cliff.y_max - cliff.y_min,
                facecolor="#7f1d1d",
                edgecolor="#ef4444",
                hatch="///",
                alpha=0.45,
            )
        )

    ax.scatter(
        [arena.dock_pose.x], [arena.dock_pose.y], marker="s", s=110, color="#facc15", zorder=6
    )
    for h in arena.humans:
        ax.scatter([h.pose.x], [h.pose.y], marker="*", s=150, color="#e879f9", zorder=6)

    (gt_line,) = ax.plot([], [], color="#22c55e", linewidth=1.6, alpha=0.75, label="Ground Truth")
    (slam_line,) = ax.plot([], [], color="#38bdf8", linewidth=2.0, label="SLAM Trajectory")
    ray_lines = [ax.plot([], [], color="#ef4444", linewidth=0.55, alpha=0.38)[0] for _ in range(24)]
    robot_body = patches.Circle(
        (0.4, 0.3), radius=0.045, facecolor="#38bdf8", edgecolor="#f8fafc", lw=1.2, zorder=7
    )
    ax.add_patch(robot_body)
    (heading_line,) = ax.plot([], [], color="#facc15", linewidth=2.2, zorder=8)

    hud_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#0f172a",
            "edgecolor": "#334155",
            "alpha": 0.9,
        },
    )

    ax.set_xlim(0.0, arena.width_m)
    ax.set_ylim(0.0, arena.height_m)
    ax.set_aspect("equal")
    ax.set_title(
        "Vector Autonomy Lab — Live Log-Odds SLAM & Behavior-Tree Execution",
        fontweight="bold",
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.88)

    def _update(frame_num: int):
        probs, rec_idx, cov = frames_data[frame_num]
        rec = ep.records[rec_idx]
        im.set_data(probs)

        sl_x = [r.slam_pose.x for r in ep.records[: rec_idx + 1]]
        sl_y = [r.slam_pose.y for r in ep.records[: rec_idx + 1]]
        gt_x = [r.gt_pose.x for r in ep.records[: rec_idx + 1]]
        gt_y = [r.gt_pose.y for r in ep.records[: rec_idx + 1]]
        slam_line.set_data(sl_x, sl_y)
        gt_line.set_data(gt_x, gt_y)

        p = rec.gt_pose
        robot_body.center = (p.x, p.y)
        robot_body.set_facecolor(_BRANCH_COLORS.get(rec.active_branch, "#38bdf8"))
        heading_line.set_data(
            [p.x, p.x + 0.09 * math.cos(p.theta)],
            [p.y, p.y + 0.09 * math.sin(p.theta)],
        )

        scan = arena.cast_range_scan(
            pose=p,
            timestamp_s=rec.time_s,
            num_rays=24,
            fov_rad=2.0 * math.pi,
            max_range_m=1.6,
            range_noise_std_m=0.0,
        )
        for rl, ang, dist in zip(ray_lines, scan.angles_rad, scan.ranges_m, strict=True):
            wx = p.x + float(dist) * math.cos(p.theta + float(ang))
            wy = p.y + float(dist) * math.sin(p.theta + float(ang))
            rl.set_data([p.x, wx], [p.y, wy])

        hud_text.set_text(
            f"t = {rec.time_s:4.1f}s | BT: {rec.active_branch:<14s} | "
            f"Bat: {rec.battery_pct:5.1f}% | Map Cov: {cov:4.1f}%"
        )
        return [im, gt_line, slam_line, heading_line, robot_body, hud_text, *ray_lines]

    anim = animation.FuncAnimation(fig, _update, frames=len(frames_data), interval=66, blit=False)
    anim.save(out_path, writer=animation.PillowWriter(fps=15))
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proof_dir = repo_root / "docs" / "proof"
    proof_dir.mkdir(parents=True, exist_ok=True)

    runner = SimulationEpisodeRunner(arena=Arena2D.default_lab_arena(), dt_s=0.1)

    ep1 = runner.run_episode(
        "ep1_full_autonomy",
        num_steps=640,
        initial_pose=Pose2D(0.42, 0.30, 0.0),
        initial_battery_pct=37.5,
        low_battery_threshold_pct=25.0,
        odom_noise=OdometryNoiseParams.nominal(),
        use_scan_matching=True,
        scheduled_intents={
            80: "Navigate to x=1.14, y=1.34",
            440: "Give me a battery status report",
        },
        seed=42,
    )

    ep2_cliff = runner.run_episode(
        "ep2_cliff_recovery",
        num_steps=260,
        initial_pose=Pose2D(1.88, 0.44, -math.pi / 4.0),
        initial_battery_pct=88.0,
        odom_noise=OdometryNoiseParams.nominal(),
        use_scan_matching=True,
        patrol_waypoints=[
            Pose2D(2.12, 0.22, -math.pi / 2.0),
            Pose2D(1.94, 0.88, math.pi / 2.0),
            Pose2D(2.05, 1.48, math.pi),
        ],
        seed=7,
    )

    ideal = runner.run_episode(
        "ablation_ideal",
        num_steps=340,
        initial_battery_pct=90.0,
        odom_noise=OdometryNoiseParams(1.0, 1.0, 0.0, 0.0),
        use_scan_matching=False,
        seed=11,
    )
    nom_no_slam = runner.run_episode(
        "ablation_nom_no_slam",
        num_steps=340,
        initial_battery_pct=90.0,
        odom_noise=OdometryNoiseParams.nominal(),
        use_scan_matching=False,
        seed=11,
    )
    nom_slam = runner.run_episode(
        "ablation_nom_slam",
        num_steps=340,
        initial_battery_pct=90.0,
        odom_noise=OdometryNoiseParams.nominal(),
        use_scan_matching=True,
        seed=11,
    )
    high_no_slam = runner.run_episode(
        "ablation_high_no_slam",
        num_steps=340,
        initial_battery_pct=90.0,
        odom_noise=OdometryNoiseParams.high_slip(),
        use_scan_matching=False,
        seed=11,
    )
    high_slam = runner.run_episode(
        "ablation_high_slam",
        num_steps=340,
        initial_battery_pct=90.0,
        odom_noise=OdometryNoiseParams.high_slip(),
        use_scan_matching=True,
        seed=11,
    )

    plot_slam_occupancy_trajectory(ep1, proof_dir / "slam_occupancy_trajectory.png")
    plot_kinematics_odometry_error(
        ideal,
        nom_no_slam,
        nom_slam,
        high_no_slam,
        high_slam,
        proof_dir / "kinematics_odometry_error.png",
    )
    plot_behavior_tree_timeline(ep1, ep2_cliff, proof_dir / "behavior_tree_timeline.png")
    render_slam_exploration_gif(ep1, proof_dir / "slam_exploration.gif")

    branch_counts: dict[str, int] = {}
    for rec in ep1.records:
        branch_counts[rec.active_branch] = branch_counts.get(rec.active_branch, 0) + 1

    metrics_summary = {
        "episode_1_full_autonomy": {
            "steps": len(ep1.records),
            "duration_s": ep1.records[-1].time_s,
            "final_battery_pct": ep1.records[-1].battery_pct,
            "slam_metrics": ep1.slam_metrics,
            "mean_odom_pos_error_cm": round(ep1.mean_odom_error_m * 100.0, 3),
            "mean_slam_pos_error_cm": round(ep1.mean_slam_error_m * 100.0, 3),
            "branch_tick_counts": branch_counts,
            "events": ep1.blackboard.event_log,
        },
        "episode_2_cliff_recovery": {
            "steps": len(ep2_cliff.records),
            "cliff_ticks": sum(1 for r in ep2_cliff.records if r.active_branch == "CLIFF_AVOID"),
            "events": ep2_cliff.blackboard.event_log,
        },
        "ablation_odometry_vs_slam": {
            "ideal_mean_err_cm": round(ideal.mean_odom_error_m * 100.0, 3),
            "nominal_raw_odom_mean_err_cm": round(nom_no_slam.mean_odom_error_m * 100.0, 3),
            "nominal_slam_mean_err_cm": round(nom_slam.mean_slam_error_m * 100.0, 3),
            "high_slip_raw_odom_mean_err_cm": round(high_no_slam.mean_odom_error_m * 100.0, 3),
            "high_slip_slam_mean_err_cm": round(high_slam.mean_slam_error_m * 100.0, 3),
            "nominal_max_odom_err_cm": round(nom_no_slam.max_odom_error_m * 100.0, 3),
            "nominal_max_slam_err_cm": round(nom_slam.max_slam_error_m * 100.0, 3),
            "high_slip_max_odom_err_cm": round(high_no_slam.max_odom_error_m * 100.0, 3),
            "high_slip_max_slam_err_cm": round(high_slam.max_slam_error_m * 100.0, 3),
        },
    }

    (proof_dir / "simulation_metrics.json").write_text(
        json.dumps(metrics_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
