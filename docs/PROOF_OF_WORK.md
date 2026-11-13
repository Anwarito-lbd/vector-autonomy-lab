# Proof of Work: Verification Logs & Artifact Audit

> **Simulation Truthfulness Notice**: All trajectories, SLAM occupancy metrics, battery curves, and Behavior-Tree traces below were produced by deterministic software simulations using `SimulatedVectorRobot` (`src/vector_lab/hal/simulated.py`), not live physical DDL Vector hardware. The generated hero visual (`docs/images/vector-lab-hero.png`) is explicitly labeled as an AI-generated architectural concept schematic.

---

## 1. Environment & Tooling Inventory

| Component | Version |
| --- | --- |
| Python | `3.12.10` |
| `pytest` | `9.1.1` |
| `ruff` | `0.16.8` |
| `numpy` | `2.5.1` |
| `matplotlib` | `3.11.2` |
| `Pillow` | `12.3.0` |

---

## 2. Test Suite Verification (`pytest -v`)

Executed after final source edits (`13 passed in 5.27s`, zero skipped tests, zero warnings):

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\issao\Downloads\portfolio\agy-work\vector-autonomy-lab
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.14.2
collected 13 items

tests\test_behavior_tree.py ..                                           [ 15%]
tests\test_closed_loop_episode.py .                                      [ 23%]
tests\test_hal_and_perception.py ...                                     [ 46%]
tests\test_kinematics.py ....                                            [ 76%]
tests\test_slam_and_planner.py ...                                       [100%]

============================= 13 passed in 5.27s ==============================
```

### Coverage by Subsystem

- `tests/test_kinematics.py` (`4` tests): `twist_to_wheels` / `wheels_to_twist` exact round-trip & curvature-preserving saturation; exact $2\pi$ ICC circular loop-closure invariant; straight-line heading invariance; deterministic seeded noisy odometry & systematic bias direction.
- `tests/test_slam_and_planner.py` (`3` tests): Bresenham line rasterization endpoints & 8-connectivity; Bayesian log-odds clamping `[l_min, l_max]` & probability updates; A* collision-free routing around central pillar with C-space inflation.
- `tests/test_behavior_tree.py` (`2` tests): `Sequence`, `Selector`, `Parallel`, `Inverter`, reactive preemption; 6-level `VectorAutonomyRoot` preemption (`CLIFF_AVOID > OPERATOR_STOP > OBSTACLE_AVOID > DOCK_CHARGE > VOICE_INTENT > FACE_GREET > PATROL`).
- `tests/test_hal_and_perception.py` (`3` tests): `create_robot_hal` factory switching, `WirePodGRPCAdapter` mm/s `DriveWheels` payload & Wire-Pod URL/state parsing, `SimulatedVectorRobot` connection guard & cliff detection, `VisionProcessor` Sobel edge density, `DeterministicIntentParser` & `LLMIntentAdapter` offline fallback.
- `tests/test_closed_loop_episode.py` (`1` test): Full 560-step closed-loop episode verifying SLAM coverage, scan-matched localization improvement over raw odometry, `VOICE_GOTO`, `FACE_GREET`, and autonomous `DOCK_CHARGE` recharge.

---

## 3. Static Linting (`ruff check src tests scripts`)

```text
All checks passed!
```

---

## 4. Multi-Episode Deterministic Simulation Outputs (`scripts/run_simulations.py`)

Measured JSON summary saved to [`docs/proof/simulation_metrics.json`](proof/simulation_metrics.json):

- **Episode 1 (`ep1_full_autonomy`, 640 ticks / 63.9 s)**:
  - Explored map coverage: `91.87%`
  - Occupied-cell precision / recall (observed) / IoU (observed): `0.8993` / `0.8283` / `0.758` (`661` TP, `74` FP cells)
  - Mean raw odometry error: `165.833 cm` vs. Mean SLAM scan-matched error: `0.705 cm`
  - Final battery after autonomous dock return & charging: `83.79%`
  - Branch tick distribution: `PATROL: 290`, `FACE_GREET: 12`, `VOICE_GOTO: 33`, `DOCK_CHARGE: 303`, `OBSTACLE_AVOID: 2`
- **Episode 2 (`ep2_cliff_recovery`, 260 ticks / 26.0 s)**:
  - `cliff_ticks`: `12` (two 6-tick cliff-detection & reverse-spin recoveries on the SE table cutout, skipping the hazardous waypoint and completing corridor patrol)
- **Ablation Study (340 ticks / 34.0 s)**:
  - `ideal_mean_err_cm`: `0.0`
  - `nominal_raw_odom_mean_err_cm`: `49.997` vs. `nominal_slam_mean_err_cm`: `1.106` (max `94.905` vs. `1.607 cm`)
  - `high_slip_raw_odom_mean_err_cm`: `61.142` vs. `high_slip_slam_mean_err_cm`: `1.209` (max `98.944` vs. `3.561 cm`)

---

## 5. Generated Visual & Data Proof Artifacts

| Artifact Path | Size (Bytes) | Description |
| --- | ---: | --- |
| `docs/images/vector-lab-hero.png` | `1,347,308` | Concept schematic illustration (`image-generator`) |
| `docs/proof/slam_occupancy_trajectory.png` | `244,727` | Ground-truth vs. odometry vs. SLAM + occupancy grid & A* path |
| `docs/proof/kinematics_odometry_error.png` | `247,110` | Position & heading drift curves and slip ablation bar chart |
| `docs/proof/behavior_tree_timeline.png` | `331,218` | BT branch preemption timeline, battery/ToF triggers & cliff recovery |
| `docs/proof/slam_exploration.gif` | `4,944,108` | Animated 15 fps live log-odds SLAM exploration & ToF ray visualization |
| `docs/proof/simulation_metrics.json` | `3,211` | Exact machine-readable metrics from `scripts/run_simulations.py` |
