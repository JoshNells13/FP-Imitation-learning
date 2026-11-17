# Technical Report: Humanoid Whole-Body Motion Imitation Learning via GMR, BeyondMimic, and MJLab (MuJoCo)

---

## 1. System Overview

This project implements an end-to-end humanoid whole-body motion imitation learning system on the **29-DoF Unitree G1** (`g1_29dof_rev_1_0`) strictly within **MuJoCo** and **MJLab**, without any dependency on ROS 2, Isaac Sim, or Isaac Lab.

```
Human Motion Capture (.bvh: Xsens / LAFAN1)
        │
        ▼
[Stage 1: GMR (General Motion Retargeting)]
  - Human-to-robot body scale calibration (actual_human_height)
  - Two-stage Mink + DAQP configuration-space differential IK
  - Automatic foot-ground Z-contact alignment
        │
        ▼
Retargeted Robot Motion (.pkl & 36-col .csv @ 30 Hz)
        │
        ▼
[Stage 2: MJLab Forward Kinematics & Interpolation]
  - Linear & Slerp interpolation from 30 Hz to 50 Hz (199 frames)
  - Finite-difference base and joint velocities
  - MuJoCo-Warp forward kinematics (sim.forward()) across all 30 bodies
        │
        ▼
MJLab Reference Trajectory (.npz @ 50 Hz)
        │
        ▼
[Stage 3: BeyondMimic PPO Training in MJLab]
  - 2,048 GPU-parallel MuJoCo-Warp environments (~40,755 steps/s)
  - Anchor-yaw-aligned relative keybody tracking (torso_link anchor, 14 keybodies)
  - Asymmetric Actor (160-D) / Privileged Critic (286-D) MLP [512, 256, 128]
  - 6 exponential tracking rewards + 3 smoothness/contact regularizers
  - Adaptive failure-bin curriculum sampling + Reference State Initialization (RSI)
        │
        ▼
Trained Policy (.pt & .onnx) + Quantitative Evaluation + Simulation Video (.mp4)
```

---

## 2. Environment Setup

| Component | Version / Specification |
|---|---|
| **OS** | Windows 11 (x86_64) |
| **GPU** | NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM, Driver 596.13, CUDA 13.2) |
| **Python** | CPython `3.12.16` (`.venv\Scripts\python.exe`) |
| **PyTorch** | `2.14.1+cu130` |
| **MuJoCo / MuJoCo-Warp** | `mujoco==3.16.0`, `mujoco-warp==3.16.0`, `warp-lang==1.19.0` |
| **MJLab** | `1.6.0` (`third_party/mjlab`) |
| **GMR** | `0.2.0` (`third_party/GMR`, `mink==1.4.0`, `qpsolvers==4.13.0`, `daqp==0.11.0`) |
| **BeyondMimic Reference** | `HybridRobotics/whole_body_tracking` (`third_party/whole_body_tracking`) |
| **RL Library** | `rsl-rl-lib==5.5.2`, `tensorboard==2.21.0` |

### Key Environment & Platform Engineering Fixes
1. **DA-QP Solver Selection on Windows**: Upstream `GMR` specifies `qpsolvers[proxqp]`, which attempts a source build of `proxsuite` that fails on Windows (`cmeel` Poetry key error). Because GMR's `GeneralMotionRetargeting` solver defaults to `solver="daqp"`, prebuilt `daqp==0.11.0` wheels were used directly.
2. **Headless Xsens BVH Parsing**: `general_motion_retargeting/utils/xsens.py` imports `OffsetManager` from `CurveEditor.py`, which imports `PyQt6` at module load time. In [`scripts/retarget_bvh_gmr.py`](file:///c:/Users/ASUS/Documents/human-imitation-learning/scripts/retarget_bvh_gmr.py), a lightweight headless `OffsetManager` stub is registered prior to import so both `bvh_xsens` and `bvh_lafan1` pipelines execute headlessly without GUI dependencies.
3. **MuJoCo-Warp CUDA Graph Overflow Print Suppression**: In `mujoco-warp==3.16.0`, `wp_opt.warn_overflow` defaults to non-zero, emitting GPU-side `printf` warnings inside captured CUDA graphs when any environment reaches the solver iteration cap. Setting `wp_opt.warn_overflow = 0` in [`third_party/mjlab/src/mjlab/sim/sim.py`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/sim/sim.py#L193-L198) prior to CUDA graph capture increased PPO throughput to **40,755 steps/second** on the RTX 5070 Laptop GPU.

---

## 3. Motion Retargeting Pipeline (GMR)

Human motion capture data in `.bvh` format is retargeted to the 29-DoF Unitree G1 (`third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml`) using [`scripts/retarget_bvh_gmr.py`](file:///c:/Users/ASUS/Documents/human-imitation-learning/scripts/retarget_bvh_gmr.py):

1. **Skeletal Parsing & Height Normalization**:
   - **Xsens BVH (`bvh_xsens`)**: Tested on `251021_04_boxing_120Hz_cm_3DsMax.bvh` (frames `0..480` at 120 Hz, downsampled $4\times$ to `120` frames at `30 FPS`, duration `4.00 s`, estimated human height $1.777\text{ m}$).
   - **LAFAN1 BVH (`bvh_lafan1`)**: Tested on `data/bvh/lafan1_sample.bvh` (`90` frames at `30 FPS`, duration `3.00 s`, estimated human height $1.750\text{ m}$).
2. **Two-Stage Differential IK (Mink + DAQP)**:
   - Stage 1 solves `ik_match_table1` (weighted root, torso, hip, knee, ankle, shoulder, elbow, and hand frame tasks) with `ConfigurationLimit` and `VelocityLimit`.
   - Stage 2 refines the configuration using `ik_match_table2` with damping $\lambda = 0.5$.
3. **Ground Contact Z-Alignment**:
   - Evaluates forward kinematics across all frames and shifts root height $z$ so the 5th-percentile lowest `ankle_roll_link` height sits at $z = 0.038\text{ m}$ (matching the G1 foot sole thickness).
4. **Quaternion Convention Alignment**:
   - MuJoCo `qpos[3:7]` uses `wxyz` (`[qw, qx, qy, qz]`), whereas MJLab's `MotionLoader` expects `xyzw` (`[qx, qy, qz, qw]`) in columns `3:7` of the 36-column CSV. [`scripts/retarget_bvh_gmr.py`](file:///c:/Users/ASUS/Documents/human-imitation-learning/scripts/retarget_bvh_gmr.py) explicitly permutes `qpos[3:7][[1, 2, 3, 0]]` so downstream conversion is exact.

---

## 4. Motion Reference Format (`.csv` $\rightarrow$ `.npz`)

Implemented in [`scripts/convert_csv_to_npz.py`](file:///c:/Users/ASUS/Documents/human-imitation-learning/scripts/convert_csv_to_npz.py):
- **Input**: `data/retargeted/g1_boxing.csv` (`120` frames $\times$ `36` columns at `30 FPS`).
- **Interpolation**: Linear interpolation for root position and 29 joint angles; spherical linear interpolation (`slerp`) for root orientation from `30 FPS` to the policy control frequency of `50 FPS` (`199` frames, `3.98 s`).
- **Forward Kinematics Rollout**: Executes `sim.forward()` inside MJLab's Unitree G1 scene (`unitree_g1_flat_tracking_env_cfg`) to record world-frame link poses and velocities for all 30 robot bodies (`world` + 29 articulated links).
- **Output (`data/motions/g1_boxing.npz`)**:
  - `fps`: `(1,)` = `[50]`
  - `joint_pos`: `(199, 29)`, `float32`
  - `joint_vel`: `(199, 29)`, `float32`
  - `body_pos_w`: `(199, 30, 3)`, `float32`
  - `body_quat_w`: `(199, 30, 4)`, `float32` (`wxyz`)
  - `body_lin_vel_w`: `(199, 30, 3)`, `float32`
  - `body_ang_vel_w`: `(199, 30, 3)`, `float32`

---

## 5. BeyondMimic Methodology & Adaptation

Full mathematical details and side-by-side repository comparisons are provided in [`docs/compatibility_and_porting.md`](file:///c:/Users/ASUS/Documents/human-imitation-learning/docs/compatibility_and_porting.md). The core BeyondMimic components preserved in `Mjlab-Tracking-Flat-Unitree-G1` are:

1. **Anchor-Yaw-Aligned Relative Keybody Representation**:
   - Anchor body: `torso_link`.
   - 14 tracked keybodies: `pelvis`, `left/right_hip_roll_link`, `left/right_knee_link`, `left/right_ankle_roll_link`, `torso_link`, `left/right_shoulder_roll_link`, `left/right_elbow_link`, `left/right_wrist_yaw_link`.
   - Reference keybody positions and orientations are transformed into the robot's current horizontal anchor frame via `yaw_quat(q_anchor_sim * conj(q_anchor_ref))` so tracking rewards and critic observations remain invariant to global XY drift and yaw offset.
2. **Asymmetric Actor–Critic Observations**:
   - **Actor (`160` dims)**: `command` `[q_ref, dq_ref]` (58) + `motion_anchor_pos_b` (3) + `motion_anchor_ori_b` 6D rotation (6) + `base_ang_vel` (3) + `joint_pos_rel` (29) + `joint_vel_rel` (29) + `last_action` (29), with additive sensor noise and empirical normalization.
   - **Critic (`286` dims)**: Adds noise-free `body_pos_b` ($14 \times 3 = 42$), `body_ori_b` ($14 \times 6 = 84$), and `base_lin_vel` (3).
3. **Reward Structure**:
   - 6 exponential tracking rewards (`motion_global_root_pos` $w=0.5,\sigma=0.3$; `motion_global_root_ori` $w=0.5,\sigma=0.4$; `motion_body_pos` $w=1.0,\sigma=0.3$; `motion_body_ori` $w=1.0,\sigma=0.4$; `motion_body_lin_vel` $w=1.0,\sigma=1.0$; `motion_body_ang_vel` $w=1.0,\sigma=3.14$).
   - 3 regularizers (`action_rate_l2` $w=-0.1$; `joint_limit` $w=-10.0$; `self_collisions` $w=-10.0$).
4. **Adaptive Failure-Bin Curriculum Sampling & RSI**:
   - Bins the trajectory into 1-second segments, smooths per-bin termination counts with causal exponential kernel $k e^{-k i}$ ($k=0.8$), and mixes with uniform sampling ($\alpha_{\text{uniform}}=0.1$).
5. **Task-Space Height & Gravity Terminations**:
   - Terminates early if $|z_{\text{anchor}}^{\text{ref}} - z_{\text{anchor}}^{\text{sim}}| > 0.25\text{ m}$, $\max_{e \in \mathcal{E}} |z_e^{\text{ref}} - z_e^{\text{sim}}| > 0.25\text{ m}$ ($\mathcal{E} = \{\text{ankles, wrists}\}$), or projected gravity $|g_z^{\text{ref}} - g_z^{\text{sim}}| > 0.8$.

---

## 6. PPO Training Configuration

Configured in [`scripts/train_ppo.py`](file:///c:/Users/ASUS/Documents/human-imitation-learning/scripts/train_ppo.py) and [`rl_cfg.py`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/config/g1/rl_cfg.py):

| Hyperparameter | Value |
|---|---|
| **Parallel Environments (`num_envs`)** | `2,048` |
| **Physics Timestep / Control Frequency** | `200 Hz` ($\Delta t = 0.005\text{ s}$), decimation `4` $\rightarrow$ `50 Hz` ($\Delta t_{\text{ctrl}} = 0.02\text{ s}$) |
| **Horizon per Iteration (`num_steps_per_env`)** | `24` (`49,152` transitions / iteration) |
| **Total Iterations / Total Steps** | `500` iterations (`24,576,000` environment steps) |
| **Actor & Critic Hidden Dims** | `[512, 256, 128]`, activation `ELU`, `actor/critic_obs_normalization = True` |
| **PPO Clip Parameter ($\epsilon$)** | `0.2` |
| **Discount ($\gamma$) / GAE ($\lambda$)** | `0.99` / `0.95` |
| **Learning Epochs / Mini-batches** | `5` epochs, `4` mini-batches |
| **Learning Rate / KL Schedule** | `1.0e-3` (adaptive KL target `0.01`) |
| **Entropy Coefficient** | `0.005` |
| **Training Speed / Wall-Clock Time** | `~40,755 steps/sec` (`8 min 56 sec` total on RTX 5070 Laptop GPU) |

---

## 7. Experimental Results

### 7.1 Training Convergence Summary (`logs/rsl_rl/g1_tracking/2026-11-09_11-40-34_beyondmimic_g1_boxing`)
- **Mean Episode Return**: Increased from **`-4.19`** (iteration 0) to **`+21.47`** (iteration 498) / **`+20.33`** (iteration 499).
- **Mean Episode Length**: Increased from **`23.72` steps** ($0.47\text{ s}$) at iteration 0 to **`474.60` steps** ($9.49\text{ s}$ out of $10.0\text{ s}$ max episode horizon) at iteration 498.
- **Training Relative Keybody Error (`error_body_pos`)**: Decreased from **`1.748 m`** (iteration 0) to **`0.090 m`** (iteration 498).
- **End-Effector Termination Rate (`ee_body_pos`)**: Decreased from **`198.71`** early in training to **`0.22–0.50`** by iterations 460–499.
- Training curves are saved at [`outputs/evaluation/ppo_training_curves.png`](file:///c:/Users/ASUS/Documents/human-imitation-learning/outputs/evaluation/ppo_training_curves.png).

### 7.2 Quantitative Evaluation Metrics ([`outputs/evaluation/metrics.json`](file:///c:/Users/ASUS/Documents/human-imitation-learning/outputs/evaluation/metrics.json))
Evaluated via [`scripts/evaluate_and_play.py`](file:///c:/Users/ASUS/Documents/human-imitation-learning/scripts/evaluate_and_play.py) over the full `199`-frame (`3.98 s`) boxing trajectory and across `256` perturbed parallel episodes:

| Evaluation Metric | Value |
|---|---|
| **Joint Position Tracking MAE (29 DoFs)** | **`0.1504 rad` (`8.62°`)** |
| **Joint Position Tracking RMSE (29 DoFs)** | **`0.1935 rad` (`11.08°`)** |
| **Perturbed Batch Joint MAE (256 envs)** | **`0.1514 rad` (`8.67°`)** |
| **Perturbed Batch Joint RMSE (256 envs)** | **`0.1982 rad` (`11.35°`)** |
| **Joint Velocity Tracking MAE** | **`0.2767 rad/s`** |
| **Joint Velocity Tracking RMSE** | **`0.5852 rad/s`** |
| **World-Frame Keybody Error (MPKPE, 14 bodies)** | **`0.0737 m` (`73.68 mm`)** |
| **Anchor-Relative Keybody Error (R-MPKPE, 14 bodies)** | **`0.0550 m` (`55.00 mm`)** |
| **Hand End-Effector Error (`left/right_wrist_yaw_link`)** | **`0.0961 m` (`96.15 mm`)** |
| **Foot End-Effector Error (`left/right_ankle_roll_link`)** | **`0.1023 m` (`102.34 mm`)** |
| **Torso Anchor Orientation Error (`torso_link`)** | **`0.0658 rad` (`3.77°`)** |
| **Motion Completion Success Rate (256 episodes)** | **`94.92%` (`243 / 256` episodes)** |

Per-joint reference vs. simulated trajectory plots are saved at [`outputs/evaluation/joint_tracking_comparison.png`](file:///c:/Users/ASUS/Documents/human-imitation-learning/outputs/evaluation/joint_tracking_comparison.png).

### 7.3 Recorded Simulation Videos
1. **GMR Retargeted Motion Preview**: [`outputs/videos/gmr_retargeted_boxing.mp4`](file:///c:/Users/ASUS/Documents/human-imitation-learning/outputs/videos/gmr_retargeted_boxing.mp4)
2. **MJLab 50 FPS Reference Trajectory**: [`outputs/videos/reference_motion.mp4`](file:///c:/Users/ASUS/Documents/human-imitation-learning/outputs/videos/reference_motion.mp4)
3. **Trained BeyondMimic PPO Policy Tracking in MuJoCo**: [`outputs/videos/trained_policy_tracking.mp4`](file:///c:/Users/ASUS/Documents/human-imitation-learning/outputs/videos/trained_policy_tracking.mp4)

---

## 8. Limitations and Future Work

1. **Single-Clip vs. Multi-Clip Motion Prior**: The current policy is trained on a single retargeted skill (`g1_boxing.npz`). Extending `MotionCommand` to index a multi-clip motion bank with phase/clip embeddings or a variational motion latent space will enable multi-skill generalization.
2. **Flat-Terrain Assumption**: Training and evaluation are performed on flat ground (`unitree_g1_flat_tracking_env_cfg`). Incorporating heightfield terrain randomization and foot-clearance rewards would improve robustness on uneven surfaces.
3. **Extended Training Horizon**: Running 1,500–3,000 PPO iterations further tightens lower-body hip pitch tracking error (`0.40–0.50 rad` currently used by the policy to maintain dynamic balance under domain randomization) toward `< 0.10 rad`.

