# Compatibility Investigation & BeyondMimic-to-MJLab Porting Architecture

## 1. Executive Summary

This document presents the technical compatibility analysis and porting methodology connecting:
1. **BeyondMimic** ([`HybridRobotics/whole_body_tracking`](https://github.com/HybridRobotics/whole_body_tracking)) — The reference whole-body humanoid motion tracking methodology.
2. **MJLab** ([`mujocolab/mjlab`](https://github.com/mujocolab/mjlab)) — The GPU-accelerated MuJoCo-Warp reinforcement learning framework adopting an Isaac Lab-style manager API.
3. **GMR (General Motion Retargeting)** ([`YanjieZe/GMR`](https://github.com/YanjieZe/GMR)) — Two-stage Mink/MuJoCo differential inverse kinematics retargeting from human motion capture (BVH) to humanoid robot embodiments.
4. **Unitree G1 (29-DoF)** — The target humanoid robot model (`g1_29dof_rev_1_0`, 12 leg DoFs + 3 waist DoFs + 14 arm DoFs).

As noted in the upstream BeyondMimic repository (`whole_body_tracking/README.md`), the original Isaac Lab implementation has been officially retired in favor of **MJLab's tracking task** (`src/mjlab/tasks/tracking`). However, integrating **GMR -> Motion Conversion -> MJLab BeyondMimic PPO Training -> Evaluation & Video Playback** on a local workstation without Isaac Sim, ROS 2, or mandatory Weights & Biases (W&B) cloud dependencies requires resolving several critical interface, coordinate convention, and platform compatibility differences documented below.

---

## 2. Eleven-Point Repository & Stack Compatibility Comparison

| Dimension | Original BeyondMimic (`whole_body_tracking`) | MJLab Adaptation (`mjlab/tasks/tracking` + Local Pipeline) | GMR (`general_motion_retargeting`) | Compatibility & Adaptation Action |
| :--- | :--- | :--- | :--- | :--- |
| **1. Simulation APIs** | NVIDIA Isaac Sim 4.5 + Isaac Lab 2.1 (`isaaclab.sim`, PhysX GPU tensor API, `InteractiveScene`, `Articulation`). | MuJoCo 3.16 + `mujoco-warp` GPU backend (`mjlab.sim.Simulation`, `mjlab.scene.Scene`, `mjlab.entity.Entity`). | CPU MuJoCo (`mujoco.MjModel`, `mujoco.MjData`) + `mink.Configuration` for per-frame differential IK. | **Adapted**: Replaced PhysX articulation calls with MJLab `Entity.write_root_state_to_sim`, `write_joint_state_to_sim`, and explicit `sim.forward()` kinematic refreshes. |
| **2. Robot Model Definitions** | URDF (`unitree_description/urdf/g1/main.urdf`) converted to USD via `UrdfFileCfg(replace_cylinders_with_capsules=True)` and `ImplicitActuatorCfg`. | Native MJCF (`src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml`) compiled via `mujoco.MjSpec` with capsule foot/body collisions (`FULL_COLLISION`) and `BuiltinPositionActuatorCfg`. | Native MJCF (`assets/unitree_g1/g1_mocap_29dof.xml`) sharing identical link offsets, inertias, and 29 hinge joint axes. | **Compatible**: Both GMR and MJLab use `g1_29dof_rev_1_0` kinematics (`pelvis` at `z=0.793m`, identical 29 joints). Motor reflected inertias (`ARMATURE_5020`, `7520_14`, `7520_22`, `4010`), natural frequency ($10\text{ Hz}$), and damping ratio ($\zeta=2.0$) match exactly. |
| **3. Environment Configuration** | `@configclass TrackingEnvCfg(ManagerBasedRLEnvCfg)` with `decimation=4`, `sim.dt=0.005` ($50\text{ Hz}$ policy), `episode_length_s=10.0`. | Dataclass builder `make_tracking_env_cfg()` + `unitree_g1_flat_tracking_env_cfg()` with `decimation=4`, `timestep=0.005` ($50\text{ Hz}$ policy), `episode_length_s=10.0`. | Operates offline at source BVH frame rate (30 Hz LAFAN1 or 120 Hz Xsens), downsampled to 30 Hz in CSV. | **Compatible**: MJLab preserves the exact $200\text{ Hz}$ physics / $50\text{ Hz}$ control decimation and episode horizon of BeyondMimic. |
| **4. Observation Spaces** | **Policy (Actor, 160-dim)**: `command` (58), `motion_anchor_pos_b` (3), `motion_anchor_ori_b` (6, first 2 cols of rotation matrix), `base_lin_vel` (3), `base_ang_vel` (3), `joint_pos_rel` (29), `joint_vel_rel` (29), `last_action` (29).<br>**Privileged (Critic, 286-dim)**: Adds `robot_body_pos_b` (14×3=42) and `robot_body_ori_b` (14×6=84). | Identical **Actor (160-dim)** and **Critic (286-dim)** structure. Uses `builtin_sensor("robot/imu_lin_vel")` and `builtin_sensor("robot/imu_ang_vel")` for base velocities, and `joint_pos_rel(biased=True)` to simulate startup joint encoder bias. | N/A (Offline retargeter). | **Preserved**: Exact 160-dim actor and 286-dim asymmetric critic observation spaces and uniform noise corruption bounds are preserved. |
| **5. Action Spaces** | `JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)` scaled per-joint by `G1_ACTION_SCALE = 0.25 * effort_limit / stiffness`. | `JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=G1_ACTION_SCALE, use_default_offset=True)` with identical `0.25 * e / s` per-actuator scaling. | Outputs generalized coordinates `qpos` (7-DoF floating base + 29-DoF joint angles). | **Preserved**: Identical 29-DoF PD target offset action space and per-motor action scaling. |
| **6. Reward Functions** | Exponential tracking kernels $\exp(-\|\Delta\|^2 / \sigma^2)$:<br>• `motion_global_anchor_pos` ($w=0.5, \sigma=0.3$)<br>• `motion_global_anchor_ori` ($w=0.5, \sigma=0.4$)<br>• `motion_body_pos` ($w=1.0, \sigma=0.3$)<br>• `motion_body_ori` ($w=1.0, \sigma=0.4$)<br>• `motion_body_lin_vel` ($w=1.0, \sigma=1.0$)<br>• `motion_body_ang_vel` ($w=1.0, \sigma=3.14$)<br>Penalties:<br>• `action_rate_l2` ($w=-0.1$)<br>• `joint_limit` ($w=-10.0$)<br>• `undesired_contacts` ($w=-0.1$). | Identical 6 exponential tracking reward terms and weights/bandwidths ($\sigma$), plus `action_rate_l2` ($w=-0.1$) and `joint_limit` ($w=-10.0$). Replaces regex `undesired_contacts` with MuJoCo subtree `self_collision_cost` ($w=-10.0$, `force_threshold=10.0`, `history_length=4`). | Optimizes `mink.FrameTask` position and orientation costs across two IK tables (`ik_match_table1` and `ik_match_table2`). | **Adapted & Documented**: Preserves all 6 BeyondMimic exponential tracking terms and regularizers; adapts self-collision penalty to use MJLab's subtree `ContactSensor` with 4-substep force history. |
| **7. Reference Motion Handling** | `MotionLoader` loads `.npz` from W&B registry; `MotionCommand` maintains `time_steps`, computes anchor-yaw-aligned `body_pos_relative_w` and `body_quat_relative_w`, and performs adaptive failure-bin sampling with a **causal** 1D conv kernel (`adaptive_kernel_size=3`). | `MotionLoader` loads local `.npz` (`--env.commands.motion.motion-file`); `MotionCommand` computes identical yaw-aligned relative poses (`update_relative_body_poses`), adds `sim.forward()` synchronization on wraparound/reset, and uses `adaptive_kernel_size=1` (or configurable kernel) with EMA failure tracking. | Produces `.pkl` dict (`root_pos`, `root_rot`, `dof_pos`, `fps`) converted to 36-col `.csv` at 30 Hz. | **Adapted**: Eliminated mandatory W&B cloud registry during `csv_to_npz` conversion, training, and evaluation so local `.npz` files work seamlessly offline. |
| **8. Joint & Body Naming / Quaternions** | 29 joints in tree order; anchor body `torso_link`; 14 tracked keybodies (`pelvis`, `left/right_hip_roll_link`, `knee_link`, `ankle_roll_link`, `torso_link`, `left/right_shoulder_roll_link`, `elbow_link`, `wrist_yaw_link`); 4 end-effectors (`ankle_roll_link`, `wrist_yaw_link`). | Identical 29 joint names, `torso_link` anchor, 14 tracked keybodies, and 4 end-effector bodies. | Identical 29 joint names and link names in `g1_mocap_29dof.xml`.<br>**CRITICAL DIFFERENCE**: `bvh_to_robot.py` saves `root_rot` as `xyzw`, whereas `xsens_bvh_to_robot.py` saves `root_rot` as `wxyz`! Meanwhile `csv_to_npz.py` converts input columns `3:7` via `[:, [3, 0, 1, 2]]` assuming `xyzw` input! | **Fixed in Pipeline**: Our unified retargeting script (`scripts/retarget_bvh_gmr.py`) enforces `xyzw` quaternion ordering in the saved `.pkl` and `.csv` for **all** BVH formats (`lafan1`, `nokov`, `xsens`), guaranteeing valid unit quaternions in `wxyz` order inside `convert_csv_to_npz.py`. |
| **9. PPO Training Interfaces** | `scripts/rsl_rl/train.py` using `rsl-rl-lib` `OnPolicyRunner` (`ActorCritic` `[512, 256, 128]` ELU, `obs_normalization=True`, `clip_param=0.2`, `lr=1e-3`, `horizon=24`), tightly coupled to W&B. | `MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner)` using `rsl-rl-lib==5.5.2` with identical PPO hyperparameters (`[512, 256, 128]` ELU, `obs_normalization=True`, `lr=1e-3`, `num_steps_per_env=24`), supporting `--agent.logger tensorboard` and local motion files. | N/A | **Adapted**: Configured `scripts/train_ppo.py` to log directly to **TensorBoard** (`runs/` / `logs/rsl_rl/`) and fixed Windows ONNX export path handling when saving checkpoints. |
| **10. Checkpoint Formats** | PyTorch `.pt` (`model_state_dict`, `optimizer_state_dict`, normalizer stats, `iter`, `infos`) + ONNX export attaching normalizer and reference motion tensors. | PyTorch `.pt` (`actor_state_dict`, `critic_state_dict`, `optimizer_state_dict`, normalizer buffers inside `Actor`/`Critic` modules) + `_OnnxMotionModel` exporting `policy.onnx` with embedded reference trajectory buffers. | N/A | **Compatible**: Checkpoints save both `.pt` weights and standalone `.onnx` policies. |
| **11. Simulation Playback & Metrics** | Requires Isaac Sim Omniverse viewport or headless PhysX camera; evaluates via `play.py` requiring W&B run path. | Native MuJoCo `OffscreenRenderer` (renders robot + translucent green reference ghost mesh) + `ViserPlayViewer` / `NativeMujocoViewer`. Built-in `evaluate.py` required W&B and lacked Joint MAE / Joint RMSE. | `RobotMotionViewer` (requires interactive GLFW window by default). | **Extended**: Created `scripts/evaluate_and_play.py` to work with local checkpoints and local `.npz` files, computing **Joint MAE**, **Joint RMSE**, **MPKPE**, **R-MPKPE**, **Joint Velocity Error**, **End-Effector Errors**, and **Success Rate**, while recording side-by-side ghost-overlay simulation `.mp4` videos. |

---

## 3. Core BeyondMimic Methodology Preserved in MJLab

### 3.1 Anchor-Relative Reference Frame Alignment
A central insight of BeyondMimic is that evaluating all body positions in the global world frame overly penalizes benign horizontal translation drift, whereas evaluating purely in local joint space fails to capture end-effector and center-of-mass task-space tracking. BeyondMimic resolves this by re-anchoring the reference keybodies at every timestep using the robot's current horizontal anchor position $(x, y)$, the reference anchor height $z^{\text{ref}}$, and the yaw-only relative orientation $\Delta q_{\text{yaw}}$:

$$\Delta \mathbf{p}_w = \begin{bmatrix} p^{\text{robot}}_{\text{anchor}, x} \\ p^{\text{robot}}_{\text{anchor}, y} \\ p^{\text{ref}}_{\text{anchor}, z} \end{bmatrix}, \quad \Delta q_{\text{yaw}} = \text{yaw\_quat}\!\left(q^{\text{robot}}_{\text{anchor}} \otimes \left(q^{\text{ref}}_{\text{anchor}}\right)^{-1}\right)$$

$$\mathbf{p}^{\text{rel}}_{b, w} = \Delta \mathbf{p}_w + \Delta q_{\text{yaw}} \odot \left(\mathbf{p}^{\text{ref}}_{b, w} - \mathbf{p}^{\text{ref}}_{\text{anchor}, w}\right), \quad q^{\text{rel}}_{b, w} = \Delta q_{\text{yaw}} \otimes q^{\text{ref}}_{b, w}$$

In MJLab, this exact transformation is implemented in [`MotionCommand.update_relative_body_poses`](../third_party/mjlab/src/mjlab/tasks/tracking/mdp/commands.py#L379-L407).

### 3.2 Asymmetric Actor-Critic Observation Design
- **Actor (160-dim, deployable on hardware)**:
  - Reference joint command: $\mathbf{q}^{\text{ref}}_t \in \mathbb{R}^{29}, \dot{\mathbf{q}}^{\text{ref}}_t \in \mathbb{R}^{29}$ (58 dims)
  - Anchor pose error in robot anchor frame: $\mathbf{p}^{\text{anchor}}_b \in \mathbb{R}^3$, 6D rotation representation $\mathbf{R}^{\text{anchor}}_{b, :, :2} \in \mathbb{R}^6$ (9 dims)
  - Base IMU linear and angular velocity: $\mathbf{v}_b \in \mathbb{R}^3, \boldsymbol{\omega}_b \in \mathbb{R}^3$ (6 dims)
  - Relative joint positions (with simulated encoder bias) and velocities: $(\mathbf{q}_t - \mathbf{q}_{\text{default}}) \in \mathbb{R}^{29}, \dot{\mathbf{q}}_t \in \mathbb{R}^{29}$ (58 dims)
  - Previous policy action: $\mathbf{a}_{t-1} \in \mathbb{R}^{29}$ (29 dims)
- **Critic (286-dim, privileged training state)**:
  - Contains all noiseless actor observations (160 dims) plus the 3D positions (14 × 3 = 42 dims) and 6D orientations (14 × 6 = 84 dims) of all 14 tracked keybodies in the robot's anchor frame.

### 3.3 Reward Formulation
The total reward at each step combines 6 positive exponential task-space tracking terms and 3 regularization penalties:

$$r_t = 0.5\, r_{\text{anchor\_pos}} + 0.5\, r_{\text{anchor\_ori}} + 1.0\, r_{\text{body\_pos}} + 1.0\, r_{\text{body\_ori}} + 1.0\, r_{\text{body\_lin\_vel}} + 1.0\, r_{\text{body\_ang\_vel}} - 0.1\, \|\mathbf{a}_t - \mathbf{a}_{t-1}\|_2^2 - 10.0\, c_{\text{joint\_limit}} - 10.0\, c_{\text{self\_collision}}$$

where each tracking term uses the Gaussian-like exponential kernel:

$$r_k = \exp\!\left(-\frac{\text{MSE}_k}{\sigma_k^2}\right)$$

### 3.4 Reference State Initialization (RSI), Adaptive Sampling & Early Termination
- **RSI & Perturbations**: Episodes initialize the robot directly at a sampled frame of the reference trajectory (`_write_reference_state_to_sim`) with uniform random perturbations to root pose, root velocity, and joint angles (`pose_range`, `velocity_range`, `joint_position_range`).
- **Adaptive Sampling**: Time bins (1-second intervals) where episodes fail early due to tracking termination accumulate failure counts (`bin_failed_count` via EMA with $\alpha=0.001$), increasing sampling probability on difficult motion segments mixed with a uniform floor (`adaptive_uniform_ratio=0.1`).
- **Height-Based Task-Space Terminations**: Episodes terminate early if:
  1. Anchor vertical error exceeds $0.25\text{ m}$ (`bad_anchor_pos_z_only`).
  2. Projected gravity $z$-component error between reference and robot anchor exceeds $0.8$ (`bad_anchor_ori`).
  3. Any of the 4 end-effectors (`left/right_ankle_roll_link`, `left/right_wrist_yaw_link`) deviates vertically from its yaw-aligned reference position by more than $0.25\text{ m}$ (`bad_motion_body_pos_z_only`).

---

## 4. Documented Pipeline Adaptations for Local Windows + MJLab Execution

1. **Headless GMR Retargeting (`scripts/retarget_bvh_gmr.py`)**:
   - Upstream `GMR/scripts/bvh_to_robot.py` and `xsens_bvh_to_robot.py` unconditionally instantiate `RobotMotionViewer`, which opens an interactive GLFW window and blocks headless execution, and `xsens_bvh_to_robot.py` saves `root_rot` in `wxyz` order instead of `xyzw`.
   - Our adaptation runs `GeneralMotionRetargeting` headlessly at full CPU speed, normalizes root height so the feet contact $z=0$, enforces consistent `xyzw` quaternion output in `.pkl` and `.csv`, and optionally renders an offscreen retargeting preview video using `mujoco.Renderer`.
2. **Offline Forward-Kinematics Motion Conversion (`scripts/convert_csv_to_npz.py`)**:
   - Upstream `mjlab/scripts/csv_to_npz.py` writes to `/tmp/motion.npz` (which fails on Windows if `/tmp` does not exist) and unconditionally calls `wandb.init()` and `run.link_artifact()` to upload to a cloud W&B registry.
   - Our adaptation saves `.npz` directly to a user-specified local path (e.g., `data/motions/*.npz`), optionally renders the reference motion video to a local `.mp4` path via MJLab's `OffscreenRenderer`, and preserves the exact `MotionLoader` interpolation (30 Hz -> 50 Hz), SO(3) angular velocity finite-differencing, and MuJoCo forward kinematics (`sim.forward()`).
3. **TensorBoard PPO Training (`scripts/train_ppo.py`)**:
   - Configures `MotionTrackingOnPolicyRunner` with `logger="tensorboard"` by default so all training metrics (rewards, tracking errors, sampling entropy, policy std, learning rate) are logged locally to TensorBoard event files without requiring a W&B account.
4. **Comprehensive Standalone Evaluation & Playback (`scripts/evaluate_and_play.py`)**:
   - Upstream `mjlab/tasks/tracking/scripts/evaluate.py` requires a `wandb_run_path` to download the motion artifact and checkpoint, and only computes MPKPE, R-MPKPE, joint velocity error, and EE error.
   - Our standalone evaluator accepts local `--checkpoint-file` and `--motion-file` paths, computes **Joint Position MAE (rad & deg)**, **Joint Position RMSE (rad & deg)**, per-joint error breakdowns, **MPKPE**, **R-MPKPE**, **Joint Velocity RMSE**, **End-Effector errors**, and **Success Rate**, and records an `.mp4` simulation playback video with the reference motion ghost overlay.

