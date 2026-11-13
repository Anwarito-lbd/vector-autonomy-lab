# Engineering Case Study: Embodied Autonomy & Log-Odds SLAM for Anki / DDL Vector

## 1. System Motivation & Hardware Constraints

Anki / Digital Dream Labs (DDL) Vector is a compact (`48 mm` tread track width) tracked social robot equipped with a forward Time-of-Flight (ToF) laser rangefinder, four downward infrared cliff sensors, a `90°` horizontal-FOV head camera, wheel encoders, an IMU, and a self-docking charging base. Building reliable autonomy on this form factor presents three coupled engineering challenges:

1. **Sim-to-Real Parity**: Hardware testing requires a paired Vector + Wire-Pod/gRPC server, yet kinematics, SLAM, path planning, and Behavior-Tree logic must be tested deterministically in CI without physical hardware.
2. **Severe Wheel-Slip Odometry Drift**: Small tracked chassis experience asymmetric tread slip and encoder calibration bias (`±1.5%–4.0%`), causing open-loop dead-reckoning to diverge by more than a meter within one minute of tabletop patrol.
3. **Hard Real-Time Safety Preemption**: High-level goals (voice commands, patrol routes, human greetings) must be immediately preempted whenever a cliff drop-off, close obstacle, operator stop, or low-battery threshold occurs.

---

## 2. Hardware Abstraction Layer (HAL) Design

All higher-level perception, planning, and control modules depend solely on the abstract [`VectorRobotHAL`](../src/vector_lab/hal/base.py) interface:

- **`SimulatedVectorRobot`** ([`src/vector_lab/hal/simulated.py`](../src/vector_lab/hal/simulated.py)): Runs deterministic 10 Hz closed-loop dynamics using exact Instantaneous Center of Curvature (ICC) integration, slab ray-AABB rangefinder casting, 4-corner cliff evaluation, LiPo discharge/recharge state dynamics, and a synthetic perspective camera renderer.
- **`WirePodGRPCAdapter`** ([`src/vector_lab/hal/wirepod_grpc.py`](../src/vector_lab/hal/wirepod_grpc.py)): Translates SI unicycle/wheel commands (`m/s`) into DDL SDK `mm/s` `DriveWheels` payloads, constructs Wire-Pod HTTP intent endpoints, and normalizes gRPC `RobotState` dictionaries into `TelemetryFrame` snapshots.
- **`create_robot_hal(target)`** ([`src/vector_lab/hal/factory.py`](../src/vector_lab/hal/factory.py)): Selects the simulation backend when `target` is `None`/`"sim"` and the hardware adapter when a Vector serial/IP is supplied.

---

## 3. Kinematics & Bayesian Log-Odds SLAM

### Curvature-Preserving Wheel Saturation

When converting unicycle twists $(v, \omega)$ to wheel speeds $(v_L, v_R)$, naive element-wise clamping distorts the turning radius $R = v / \omega$ and causes path-following overshoot. `DifferentialDriveKinematics.twist_to_wheels` instead scales both wheel velocities uniformly whenever $\max(|v_L|, |v_R|) > v_{\max}$, preserving curvature invariants (verified in `tests/test_kinematics.py`).

### Why Correlative Scan Matching Is Essential

In our 63.9-second closed-loop autonomy run (`episode_1_full_autonomy`), a realistic `1.5%` left/right tread scale asymmetry plus Gaussian slip (`linear_noise_std = 0.004 m/s`, `angular_noise_std = 0.025 rad/s`) caused raw wheel odometry to accumulate a **165.833 cm** mean planar position error—effectively driving the raw odometry estimate straight through walls and off the arena map.

By pairing integer Bresenham ray-traced log-odds updates ($l_{\text{occ}} = +0.85, l_{\text{free}} = -0.35$) with `OccupancyGridSLAM.correlative_scan_match` over a local search grid ($\Delta x, \Delta y \in [-0.04, 0.04]\text{ m}$, $\Delta \theta \in [-0.06, 0.06]\text{ rad}$), mean localization error dropped to **0.705 cm** while reconstructing **91.87%** of the arena at **0.8993** occupied-cell precision.

Across a controlled 34.0-second ablation sweep:
- **Nominal slip**: Mean error dropped from **49.997 cm** (raw odom) to **1.106 cm** (SLAM scan-matched); max error dropped from **94.905 cm** to **1.607 cm**.
- **High slip (`±4.0%` bias)**: Mean error dropped from **61.142 cm** to **1.209 cm**; max error dropped from **98.944 cm** to **3.561 cm**.

---

## 4. Reactive Behavior-Tree Engineering & Lessons Learned

### Priority Preemption Without Memory Lock-In

`build_vector_autonomy_tree` uses a **reactive** (`memory=False`) top-level `Selector` so that every 100 ms control tick re-evaluates safety and energy preconditions from top priority down:

1. `CliffSafetyGuard`
2. `OperatorHaltGuard`
3. `ObstacleSafetyGuard`
4. `EnergyDockCharge`
5. `VoiceIntentDispatch`
6. `SocialFaceGreeting`
7. `AutonomousPatrol`

### Case Diagnosis: Eliminating Cliff-Recovery Waypoint Oscillation

During initial Episode 2 simulation runs near the southeast tabletop cutout (`CliffZone(1.98, 0.04, 2.34, 0.34)`), we observed an authentic failure mode: after `CliffSafetyGuard` reversed Vector away from the drop-off and returned `NodeStatus.SUCCESS`, `AutonomousPatrol` immediately resumed pursuing the *same* hazardous waypoint (`(2.12, 0.22)`), causing repeated cliff triggers (`165` cliff ticks out of `260`).

**Fix implemented in `VectorAutonomyController.action_cliff_recover`**: Upon completing the reverse-and-spin recovery sequence, the controller now invalidates any active `GOTO_POSE` intent or advances `bb.patrol_index` past the hazardous waypoint. Re-running Episode 2 reduced cliff-recovery ticks from `165` to **`12`** (two clean 6-tick recovery pulses followed by safe corridor navigation), verified directly in `docs/proof/behavior_tree_timeline.png` Panel 3.

---

## 5. Limitations & Hardware Transition Notes

- **2D Planar Slice vs. Full 3D Sensor Rig**: The simulator models Vector's horizontal ToF fan and 4-corner chassis cliff switches over rectangular 3D obstacles with elevation metadata; physical Vector uses a single narrow-cone forward VL53L0X ToF sensor, meaning physical mapping requires active yaw sweeps during patrol (`read_range_scan` degrades cleanly to a single-ray scan in `WirePodGRPCAdapter`).
- **Simulated Metrics Disclaimer**: All quantitative metrics in this repository come from deterministic runs of `SimulatedVectorRobot` and are explicitly labeled as simulated results.
