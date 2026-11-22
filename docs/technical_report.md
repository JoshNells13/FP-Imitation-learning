# Technical Report: Humanoid Whole-Body Motion Imitation Learning on Booster K1 (22-DoF) via GMR, BeyondMimic, and MJLab (MuJoCo)

---

## 1. System Overview

This project implements an end-to-end humanoid whole-body motion imitation learning system on the **22-DoF Booster K1** (`booster_k1_22dof`) strictly within **MuJoCo** and **MJLab** (`mujoco-warp`), without any dependency on ROS 2, Isaac Sim, or Isaac Lab. It is paired with a **Stateless Ephemeral In-Memory Web Monitoring & 3D Control Platform** (`monitoring-control-imitation`) built with Flask + Socket.IO and Next.js 14 + Three.js (`@react-three/fiber`).

```text
Human Motion Capture (.bvh: LAFAN1 / Xsens / Nokov / CMU / Mixamo)
        │
        ▼
[Stage 1: Universal Zero-Config GMR (General Motion Retargeting)]
  - Auto-detection of skeleton format, unit scale (cm/m), FPS, and subject height
  - Smart Active-Clip Auto-Window selection (5.0s upright high-activity window)
  - Auto-origin (X=0, Y=0) and +X heading (yaw=0) alignment
  - Two-stage Mink + DAQP configuration-space differential IK
  - Universal K1 Kinematic Safety Filter (22-DoF joint margins, arm-torso anti-collision, ground Z-contact)
        │
        ▼
Retargeted Robot Motion (.pkl & 29-col .csv @ 30 Hz)
        │
        ▼
[Stage 2: MJLab Forward Kinematics & 50 Hz Interpolation]
  - Linear & Slerp interpolation from 30 Hz to 50 Hz
  - Finite-difference base and 22-DoF joint velocities
  - MuJoCo-Warp forward kinematics (sim.forward()) across all 23 articulated links
        │
        ▼
MJLab Reference Trajectory (.npz @ 50 Hz)
        │
        ▼
[Stage 3: BeyondMimic PPO Training in MJLab (Mjlab-Tracking-Flat-Booster-K1)]
  - 2,048 GPU-parallel MuJoCo-Warp environments (~40,000+ steps/s)
  - Anchor-yaw-aligned relative keybody tracking (Trunk anchor, 14 tracked keybodies)
  - Dedicated Upper-Body / Arm Joint & Keybody Tracking Rewards + Responsive Arm Action Scale (0.45 rad)
  - Asymmetric Actor (125-D) / Privileged Critic (251-D) MLP [512, 256, 128] with ELU & Empirical Normalization
  - Adaptive failure-bin curriculum sampling + Reference State Initialization (RSI)
        │
        ▼
Trained Policy (.pt & .onnx) + Closed-Loop 3D Rollout (.npz) + Quantitative Evaluation + Web 3D Viewer
```

---

## 2. Environment & Hardware Architecture

| Component | Version / Specification |
|---|---|
| **OS** | Windows 11 (x86_64) |
| **GPU** | NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM, CUDA 13.2 / 13.4) |
| **Python** | CPython `3.12` (`.venv\Scripts\python.exe`) |
| **PyTorch** | `2.14.1+cu130` |
| **MuJoCo / MuJoCo-Warp** | `mujoco==3.16.0`, `mujoco-warp==3.16.0`, `warp-lang==1.19.0` |
| **MJLab** | `1.6.0` (`third_party/mjlab`) |
| **GMR** | `0.2.0` (`third_party/GMR`, `mink==1.4.0`, `qpsolvers==4.13.0`, `daqp==0.11.0`) |
| **BeyondMimic Reference** | `HybridRobotics/whole_body_tracking` (`third_party/whole_body_tracking`) |
| **RL Library** | `rsl-rl-lib==5.5.2`, `tensorboard==2.21.0` |
| **Web Control & 3D Platform** | Flask + Flask-SocketIO (Stateless In-Memory) & Next.js 14 + React Three Fiber + Recharts |

### Key Platform & Engineering Enhancements
1. **Universal Zero-Config BVH Retargeting Pipeline ([`scripts/retarget_bvh_gmr.py`](file:///c:/Users/ASUS/Documents/fp-ichiro/human-imitation-learning/scripts/retarget_bvh_gmr.py) & [`lafan1.py`](file:///c:/Users/ASUS/Documents/fp-ichiro/human-imitation-learning/third_party/GMR/general_motion_retargeting/utils/lafan1.py))**:
   - Automatically detects skeleton format (`LAFAN1`, `Nokov`, `CMU`, `Mixamo`, `Xsens`), coordinate scale (`cm` vs `m`), source frame rate (`Frame Time:` header downsampled to `30 FPS`), and human subject height (90th-percentile vertical head-to-foot span).
   - **Smart Active-Clip Auto-Window (`select_optimal_bvh_window`)**: Automatically scans multi-minute mocap takes to extract the optimal `5.0 s` (`150` frames @ `30 FPS`) segment with upright posture (`Hips_Z >= 72%` of standing height) and rich whole-body activity, skipping initial T-pose calibration frames and floor-sitting segments.
   - **Universal K1 Kinematic Safety Filter (`enforce_k1_kinematic_safety`)**: Clamps inward `Shoulder_Roll` (`Left_Shoulder_Roll <= 0.12 rad`, `Right_Shoulder_Roll >= -0.12 rad`) to prevent arm-into-torso self-collision, enforces a `0.015 rad` margin inside all 22 MJCF joint limits, applies a 3-tap zero-phase temporal filter `[0.2, 0.6, 0.2]`, and aligns stance feet to $z = 0.024\text{ m}$.
2. **Upper-Body & Arm Mobility Fix ([`k1_constants.py`](file:///c:/Users/ASUS/Documents/fp-ichiro/human-imitation-learning/third_party/mjlab/src/mjlab/asset_zoo/robots/booster_k1/k1_constants.py), [`rewards.py`](file:///c:/Users/ASUS/Documents/fp-ichiro/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/mdp/rewards.py), & [`env_cfgs.py`](file:///c:/Users/ASUS/Documents/fp-ichiro/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/config/k1/env_cfgs.py))**:
   - **Root Cause Identified**: Previously, `ARMATURE_ARM = 0.015` ($\text{stiffness} = 59.22\text{ N}\cdot\text{m/rad}$) with `effort_limit = 10.0` produced an arm action scale of only `K1_ACTION_SCALE = 0.25 * e / s = 0.0422 rad` ($2.4^\circ$ per unit action). Under the `action_rate_l2` penalty ($w = -0.1$), a standard Gaussian policy output $a_t \in [-2, +2]$ could only move the arms by $\pm 0.084\text{ rad}$ ($\pm 4.8^\circ$) around the default keyframe pose, causing the arms to appear frozen during dynamic simulations.
   - **Solution**:
     1. Tuned `ARMATURE_ARM = 0.005` and `effort_limit = 18.0 N*m`, and raised `K1_ACTION_SCALE` for all 8 shoulder/elbow joints (`ALeft/Right_Shoulder_Pitch`, `Left/Right_Shoulder_Roll`, `Left/Right_Elbow_Pitch`, `Left/Right_Elbow_Yaw`) to **`0.45 rad`** ($10.6\times$ larger range per action unit), allowing full $\pm 1.35\text{ rad}$ expressive arm swings without excessive action-rate penalty.
     2. Added dedicated upper-body tracking reward terms in `booster_k1_flat_tracking_env_cfg`: `motion_arm_joint_pos` (`motion_joint_pos_error_exp` on arm DoFs `2..10`, $w = 1.5, \sigma = 0.35$) and `motion_arm_body_pos` (`motion_relative_body_position_error_exp` on the 6 arm/hand links, $w = 1.2, \sigma = 0.25$).
     3. Corrected the GMR IK arm frame orientation quaternions in [`bvh_lafan1_to_k1.json`](file:///c:/Users/ASUS/Documents/fp-ichiro/human-imitation-learning/third_party/GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_k1.json) (`[0.7071, 0, 0, -0.7071]` for left arm links and `[0, -0.7071, 0.7071, 0]` for right arm links) and restricted `terminations["ee_body_pos"]` to ground-contact feet `("left_foot_link", "right_foot_link")` so high arm waves never trigger false episode termination.
3. **Manual `ENTER` Reload in Interactive MuJoCo Viewer ([`scripts/play_interactive.py`](file:///c:/Users/ASUS/Documents/fp-ichiro/human-imitation-learning/scripts/play_interactive.py))**:
   - By default, `play_interactive.py` disables automatic termination resets and clamps the reference frame index at the end of the trajectory until the user explicitly presses **`ENTER`** in the MuJoCo window (with `--auto-reset` available for continuous looping).

---

## 3. Booster K1 (22-DoF) Kinematic & Actuator Specification

Defined in [`third_party/GMR/assets/booster_k1/k1_mjlab.xml`](file:///c:/Users/ASUS/Documents/fp-ichiro/human-imitation-learning/third_party/GMR/assets/booster_k1/k1_mjlab.xml) and [`k1_constants.py`](file:///c:/Users/ASUS/Documents/fp-ichiro/human-imitation-learning/third_party/mjlab/src/mjlab/asset_zoo/robots/booster_k1/k1_constants.py):

- **23 Articulated Bodies**: `Trunk` (root/anchor, standing height $\approx 0.53\text{ m}$), `Head_1`, `Head_2`, `Left_Arm_1..3`, `left_hand_link`, `Right_Arm_1..3`, `right_hand_link`, `Left_Hip_Pitch`, `Left_Hip_Roll`, `Left_Hip_Yaw`, `Left_Shank`, `Left_Ankle_Cross`, `left_foot_link`, `Right_Hip_Pitch`, `Right_Hip_Roll`, `Right_Hip_Yaw`, `Right_Shank`, `Right_Ankle_Cross`, `right_foot_link`.
- **22 Actuated DoFs**:
  - **Head (2 DoFs, indices `0..1`)**: `AAHead_yaw`, `Head_pitch`
  - **Left Arm (4 DoFs, indices `2..5`)**: `ALeft_Shoulder_Pitch`, `Left_Shoulder_Roll`, `Left_Elbow_Pitch`, `Left_Elbow_Yaw`
  - **Right Arm (4 DoFs, indices `6..9`)**: `ARight_Shoulder_Pitch`, `Right_Shoulder_Roll`, `Right_Elbow_Pitch`, `Right_Elbow_Yaw`
  - **Left Leg (6 DoFs, indices `10..15`)**: `Left_Hip_Pitch`, `Left_Hip_Roll`, `Left_Hip_Yaw`, `Left_Knee_Pitch`, `Left_Ankle_Pitch`, `Left_Ankle_Roll`
  - **Right Leg (6 DoFs, indices `16..21`)**: `Right_Hip_Pitch`, `Right_Hip_Roll`, `Right_Hip_Yaw`, `Right_Knee_Pitch`, `Right_Ankle_Pitch`, `Right_Ankle_Roll`

---

## 4. Motion Reference Format (`.csv` $\rightarrow$ `.npz`)

Implemented in [`scripts/convert_csv_to_npz.py`](file:///c:/Users/ASUS/Documents/fp-ichiro/human-imitation-learning/scripts/convert_csv_to_npz.py):
- **Input**: 29-column CSV (`[x, y, z, qx, qy, qz, qw, j_0, ..., j_21]` at `30 FPS`).
- **Interpolation**: Linear interpolation for root position and 22 joint angles; spherical linear interpolation (`slerp`) for root orientation from `30 FPS` to the policy control frequency of `50 FPS`.
- **Forward Kinematics Rollout**: Executes `sim.forward()` inside MJLab's Booster K1 scene (`booster_k1_flat_tracking_env_cfg`) to record world-frame link poses and velocities across all 23 robot bodies.
- **Output (`data/motions/k1_<motion>.npz`)**:
  - `fps`: `(1,)` = `[50]`
  - `joint_pos`: `(T, 22)`, `float32`
  - `joint_vel`: `(T, 22)`, `float32`
  - `body_pos_w`: `(T, 23, 3)`, `float32`
  - `body_quat_w`: `(T, 23, 4)`, `float32` (`wxyz`)
  - `body_lin_vel_w`: `(T, 23, 3)`, `float32`
  - `body_ang_vel_w`: `(T, 23, 3)`, `float32`

---

## 5. BeyondMimic Methodology & Adaptation for Booster K1

1. **Anchor-Yaw-Aligned Relative Keybody Representation**:
   - Anchor body: `Trunk`.
   - 14 tracked keybodies: `Trunk`, `Head_2`, `Left/Right_Hip_Roll`, `Left/Right_Shank`, `left/right_foot_link`, `Left/Right_Arm_2`, `Left/Right_Arm_3`, `left/right_hand_link`.
2. **Asymmetric Actor–Critic Observations**:
   - **Actor (`125` dims)**: `command` `[q_ref, dq_ref]` (44) + `motion_anchor_pos_b` (3) + `motion_anchor_ori_b` 6D rotation (6) + `base_lin_vel` (3) + `base_ang_vel` (3) + `joint_pos_rel` (22) + `joint_vel_rel` (22) + `last_action` (22).
   - **Critic (`251` dims)**: Adds privileged noise-free `body_pos_b` ($14 \times 3 = 42$) and `body_ori_b` ($14 \times 6 = 84$).
3. **Reward Structure (8 Tracking Terms + 3 Regularizers)**:
   - `motion_global_root_pos` ($w=0.5, \sigma=0.3$)
   - `motion_global_root_ori` ($w=0.5, \sigma=0.4$)
   - `motion_body_pos` ($w=1.5, \sigma=0.3$)
   - `motion_body_ori` ($w=1.5, \sigma=0.4$)
   - `motion_body_lin_vel` ($w=1.0, \sigma=1.0$)
   - `motion_body_ang_vel` ($w=1.0, \sigma=3.14$)
   - `motion_arm_joint_pos` ($w=1.5, \sigma=0.35$, upper-body arm DoFs `2..10`)
   - `motion_arm_body_pos` ($w=1.2, \sigma=0.25$, 6 upper-body arm/hand links)
   - Regularizers: `action_rate_l2` ($w=-0.1$), `joint_limit` ($w=-5.0$), `self_collisions` ($w=-10.0$).

---

## 6. Experimental Results Across Retargeted Skills

| Motion Skill | Source Dataset | Frames @ 50Hz (Duration) | Mean Episode Reward | Joint Tracking MAE | Anchor-Relative Keybody Error (R-MPKPE) | Torso Orientation Error | Trajectory Completion / Success Rate |
|---|---|---|---:|---:|---:|---:|---:|
| **Martial-Arts Front Kick (`k1_kick.npz`)** | Xsens Mocap (`120 Hz`) | `149` (`2.98 s`) | `+25.10` | `0.1510 rad` (`8.65°`) | `36.8 mm` | `0.072 rad` (`4.12°`) | **`99.61%` (`255/256`)** |
| **Expressive Dance (`k1_dance.npz`)** | LAFAN1 (`dance1_subject1.bvh`) | `199` (`3.98 s`) | `+20.43` | `0.2163 rad` (`12.39°`) | `45.4 mm` | `0.137 rad` (`7.82°`) | **`73.05%` (`187/256`, 100% det.)** |
| **Human Walking (`k1_walk.npz`)** | LAFAN1 (`walk1_subject1.bvh`) | `249` (`4.98 s`) | `+22.85` | `0.1740 rad` (`9.97°`) | `41.2 mm` | `0.089 rad` (`5.10°`) | **`100%` deterministic rollout** |

---

## 7. Web Monitoring & 3D Simulation Platform Integration

The training pipeline is directly connected to the **Website Monitoring & Control Platform** ([`../monitoring-control-imitation`](file:///c:/Users/ASUS/Documents/fp-ichiro/monitoring-control-imitation)):
- **Stateless Ephemeral In-Memory Architecture**: Operates with zero database overhead (`memory://ephemeral-live-session`); simulation buffers can be reset and discarded immediately (`POST /api/simulation/discard`).
- **Live Filesystem Change Detection (`hil_filesystem_changed`)**: Computes a real-time MD5 signature across `data/bvh/**`, `data/retargeted/`, `data/motions/`, `logs/rsl_rl/k1_tracking/`, `outputs/checkpoints/`, `outputs/evaluation/`, `outputs/videos/`, and `docs/`, broadcasting updates via Socket.IO every 3 seconds.
- **Auto-Detected Motion & Policy Endpoint (`GET /api/simulation/assets`)**: Automatically discovers every `.npz` motion and `.pt` checkpoint on disk, pairs each motion with its trained policy and closed-loop 3D rollout (`<motion>_rollout_trajectory.npz`), and falls back gracefully if older files are removed.
