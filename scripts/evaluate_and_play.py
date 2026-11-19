"""Comprehensive evaluation and simulation video recording script for Booster K1 BeyondMimic policy.

Computes all required quantitative metrics from PRD Section 10:
  1. Joint Position Tracking Error:
     - Overall & per-joint MAE (rad and deg)
     - Overall & per-joint RMSE (rad and deg)
  2. Joint Velocity Tracking Error:
     - Overall MAE and RMSE (rad/s)
  3. Keybody & End-Effector Tracking Error:
     - MPKPE (World-frame Mean Per-Keybody Position Error, m and mm)
     - R-MPKPE (Anchor-relative Mean Per-Keybody Position Error, m and mm)
     - End-Effector Position Errors (Hands & Feet, m and mm)
     - Anchor / Keybody Orientation Error (rad and deg)
  4. Motion Completion / Success Rate (%):
     - Evaluated across N_eval parallel episodes over the full trajectory duration.

Also generates:
  - Simulation playback video (.mp4) of the trained policy tracking the motion
  - Reference vs. simulated joint tracking comparison plot (.png)
  - Structured JSON evaluation report (.json)
"""

from __future__ import annotations

import os
import pathlib
import sys
from dataclasses import asdict

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = WORKSPACE_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("WARP_CACHE_PATH", str(CACHE_DIR / "warp"))

for rel_path in ("third_party/mjlab/src", "third_party/GMR"):
    full_path = str(WORKSPACE_ROOT / rel_path)
    if full_path not in sys.path:
        sys.path.insert(0, full_path)

import argparse
import json
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich import print
from rich.table import Table

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.config.k1.env_cfgs import booster_k1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.k1.rl_cfg import booster_k1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.utils.lab_api.math import quat_error_magnitude


K1_22DOF_JOINT_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]


def plot_joint_comparison(
    time_sec: np.ndarray,
    ref_joints: np.ndarray,
    sim_joints: np.ndarray,
    output_plot_path: pathlib.Path,
    robot: str = "booster_k1",
) -> None:
    """Plot reference vs. simulated joint trajectories for representative lower & upper body joints."""
    selected = [
        (10, "Left Hip Pitch"),
        (13, "Left Knee Pitch"),
        (16, "Right Hip Pitch"),
        (19, "Right Knee Pitch"),
        (14, "Left Ankle Pitch"),
        (20, "Right Ankle Pitch"),
        (2, "Left Shoulder Pitch"),
        (5, "Left Elbow Yaw"),
        (6, "Right Shoulder Pitch"),
        (9, "Right Elbow Yaw"),
    ]
    title_str = "Booster K1 (22-DoF) Whole-Body Motion Tracking: GMR Reference vs. BeyondMimic Policy"

    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
    axes_flat = axes.flatten()

    for ax_idx, (j_idx, j_label) in enumerate(selected):
        ax = axes_flat[ax_idx]
        ax.plot(time_sec, ref_joints[:, j_idx], "k--", linewidth=1.8, label="Reference (GMR)")
        ax.plot(time_sec, sim_joints[:, j_idx], "C0-", linewidth=1.6, alpha=0.9, label="Simulated (BeyondMimic PPO)")
        mae_j = float(np.mean(np.abs(sim_joints[:, j_idx] - ref_joints[:, j_idx])))
        rmse_j = float(np.sqrt(np.mean((sim_joints[:, j_idx] - ref_joints[:, j_idx]) ** 2)))
        ax.set_title(f"{j_label} (MAE={mae_j:.3f} rad, RMSE={rmse_j:.3f} rad)", fontsize=10)
        ax.set_ylabel("Angle (rad)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes_flat[-2].set_xlabel("Time (s)", fontsize=10)
    axes_flat[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(
        title_str,
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=180)
    plt.close(fig)
    print(f"[green]Saved joint tracking comparison plot to {output_plot_path}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained Booster K1 (22-DoF) BeyondMimic policy and record simulation video."
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="booster_k1",
        choices=["booster_k1"],
        help="Target humanoid robot model (booster_k1).",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        required=True,
        help="Path to trained .pt model checkpoint.",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=True,
        help="Path to MJLab .npz reference motion file.",
    )
    parser.add_argument(
        "--num-eval-envs",
        type=int,
        default=256,
        help="Number of parallel environments for multi-episode statistical evaluation (default: 256).",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="outputs/videos/trained_policy_tracking.mp4",
        help="Output .mp4 path for recorded simulation playback video.",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="outputs/evaluation/metrics.json",
        help="Output .json path for evaluation metrics.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="outputs/evaluation/joint_tracking_comparison.png",
        help="Output .png path for joint tracking plot.",
    )
    parser.add_argument(
        "--output-rollout-npz",
        type=str,
        default="",
        help="Optional output .npz path for closed-loop rollout trajectory (for Web 3D Viewer).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device (default: cuda:0).",
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt_path = pathlib.Path(args.checkpoint_file)
    if not ckpt_path.is_absolute():
        ckpt_path = (WORKSPACE_ROOT / ckpt_path).resolve()
    motion_path = pathlib.Path(args.motion_file)
    if not motion_path.is_absolute():
        motion_path = (WORKSPACE_ROOT / motion_path).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion NPZ not found: {motion_path}")

    # -------------------------------------------------------------------------
    # Part 1: Single-trajectory deterministic playback + video recording
    # -------------------------------------------------------------------------
    print(f"[bold blue]1. Running deterministic full-trajectory rollout & recording video ({args.robot})...[/bold blue]")
    play_cfg = booster_k1_flat_tracking_env_cfg(play=True)
    agent_cfg = booster_k1_tracking_ppo_runner_cfg()
    joint_names_list = K1_22DOF_JOINT_NAMES
    hand_names = ("left_hand_link", "right_hand_link")
    foot_names = ("left_foot_link", "right_foot_link")
    task_id_str = "Mjlab-Tracking-Flat-Booster-K1"
    robot_model_str = "booster_k1_22dof"
    cam_dist = 1.9

    play_cfg.scene.num_envs = 1
    play_cfg.viewer.height = 720
    play_cfg.viewer.width = 960
    play_cfg.viewer.distance = cam_dist
    play_cfg.viewer.elevation = -12.0
    play_cfg.viewer.azimuth = 135.0

    motion_cmd_cfg = play_cfg.commands["motion"]
    assert isinstance(motion_cmd_cfg, MotionCommandCfg)
    motion_cmd_cfg.motion_file = str(motion_path)
    motion_cmd_cfg.sampling_mode = "start"
    # Zero pose/velocity noise for clean deterministic evaluation & video
    motion_cmd_cfg.pose_range = {}
    motion_cmd_cfg.velocity_range = {}

    play_env = ManagerBasedRlEnv(cfg=play_cfg, device=device, render_mode="rgb_array")
    play_vec_env = RslRlVecEnvWrapper(play_env)

    runner = MotionTrackingOnPolicyRunner(
        play_vec_env, asdict(agent_cfg), log_dir=None, device=device
    )
    runner.load(str(ckpt_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    cmd_term: MotionCommand = play_env.command_manager.get_term("motion")
    traj_steps = int(cmd_term.motion.time_step_total)
    control_dt = float(play_env.step_dt)
    fps = int(round(1.0 / control_dt))

    obs, _ = play_vec_env.reset()
    # Force time_steps=0 and initialize robot right on reference frame 0
    cmd_term.time_steps.zero_()
    cmd_term._resample_command(torch.tensor([0], device=device))
    obs = play_vec_env.get_observations()

    video_frames = []
    ref_q_list = []
    sim_q_list = []
    ref_dq_list = []
    sim_dq_list = []
    sim_bpos_list = []
    sim_bquat_list = []
    actions_list = []
    mpkpe_list = []
    rmpkpe_list = []
    hand_err_list = []
    foot_err_list = []
    anchor_ori_err_list = []

    body_names = list(cmd_term.cfg.body_names)
    hand_indices = [body_names.index(hand_names[0]), body_names.index(hand_names[1])]
    foot_indices = [body_names.index(foot_names[0]), body_names.index(foot_names[1])]

    with torch.no_grad():
        for step in range(traj_steps - 1):
            # Record state before step
            ref_q = cmd_term.joint_pos[0].detach().cpu().numpy().copy()
            sim_q = cmd_term.robot.data.joint_pos[0].detach().cpu().numpy().copy()
            ref_dq = cmd_term.joint_vel[0].detach().cpu().numpy().copy()
            sim_dq = cmd_term.robot.data.joint_vel[0].detach().cpu().numpy().copy()
            sim_bpos = cmd_term.robot.data.body_link_pos_w[0].detach().cpu().numpy().copy()
            sim_bquat = cmd_term.robot.data.body_link_quat_w[0].detach().cpu().numpy().copy()

            ref_q_list.append(ref_q)
            sim_q_list.append(sim_q)
            ref_dq_list.append(ref_dq)
            sim_dq_list.append(sim_dq)
            sim_bpos_list.append(sim_bpos)
            sim_bquat_list.append(sim_bquat)

            pos_err_b = torch.norm(cmd_term.body_pos_w - cmd_term.robot_body_pos_w, dim=-1)[0]
            rel_err_b = torch.norm(cmd_term.body_pos_relative_w - cmd_term.robot_body_pos_w, dim=-1)[0]
            mpkpe_list.append(float(pos_err_b.mean().item()))
            rmpkpe_list.append(float(rel_err_b.mean().item()))
            hand_err_list.append(float(rel_err_b[hand_indices].mean().item()))
            foot_err_list.append(float(rel_err_b[foot_indices].mean().item()))

            anchor_ori_err = quat_error_magnitude(
                cmd_term.anchor_quat_w, cmd_term.robot_anchor_quat_w
            )[0]
            anchor_ori_err_list.append(float(anchor_ori_err.item()))

            frame = play_env.render()
            if frame is not None:
                video_frames.append(frame)

            actions = policy(obs)
            actions_list.append(actions[0].detach().cpu().numpy().copy())
            obs, _, _, _ = play_vec_env.step(actions)

        # Record final T-th frame state for full T-frame rollout export
        sim_q_final = cmd_term.robot.data.joint_pos[0].detach().cpu().numpy().copy()
        sim_dq_final = cmd_term.robot.data.joint_vel[0].detach().cpu().numpy().copy()
        sim_bpos_final = cmd_term.robot.data.body_link_pos_w[0].detach().cpu().numpy().copy()
        sim_bquat_final = cmd_term.robot.data.body_link_quat_w[0].detach().cpu().numpy().copy()
        act_final = policy(obs)[0].detach().cpu().numpy().copy()

    play_vec_env.close()

    rollout_rel = args.output_rollout_npz or f"outputs/evaluation/{motion_path.stem}_rollout_trajectory.npz"
    rollout_npz_path = (WORKSPACE_ROOT / rollout_rel).resolve()
    rollout_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        rollout_npz_path,
        sim_joint_pos=np.stack(sim_q_list + [sim_q_final], axis=0).astype(np.float32),
        sim_joint_vel=np.stack(sim_dq_list + [sim_dq_final], axis=0).astype(np.float32),
        sim_body_pos_w=np.stack(sim_bpos_list + [sim_bpos_final], axis=0).astype(np.float32),
        sim_body_quat_w=np.stack(sim_bquat_list + [sim_bquat_final], axis=0).astype(np.float32),
        actions=np.stack(actions_list + [act_final], axis=0).astype(np.float32),
    )
    print(f"[bold green]Saved {traj_steps}-frame closed-loop rollout NPZ to {rollout_npz_path}[/bold green]")

    # Save simulation video across 2 full loops so viewers can clearly watch the motion
    video_path = (WORKSPACE_ROOT / args.output_video).resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    looped_frames = video_frames + video_frames
    imageio.mimsave(str(video_path), looped_frames, fps=fps, macro_block_size=1)
    print(
        f"[bold green]Saved simulation playback video to {video_path} "
        f"({len(looped_frames)} frames @ {fps} FPS)[/bold green]"
    )

    ref_q_arr = np.stack(ref_q_list, axis=0)
    sim_q_arr = np.stack(sim_q_list, axis=0)
    ref_dq_arr = np.stack(ref_dq_list, axis=0)
    sim_dq_arr = np.stack(sim_dq_list, axis=0)
    time_sec = np.arange(ref_q_arr.shape[0]) * control_dt

    plot_path = (WORKSPACE_ROOT / args.output_plot).resolve()
    plot_joint_comparison(time_sec, ref_q_arr, sim_q_arr, plot_path, robot=args.robot)

    # -------------------------------------------------------------------------
    # Part 2: Multi-environment statistical rollout for Success Rate & Robustness
    # -------------------------------------------------------------------------
    print(
        f"[bold blue]2. Evaluating Success Rate & tracking robustness across "
        f"{args.num_eval_envs} parallel episodes (with pose perturbations)...[/bold blue]"
    )
    eval_cfg = booster_k1_flat_tracking_env_cfg(play=False)
    eval_cfg.scene.num_envs = args.num_eval_envs
    # Disable random mid-episode velocity pushes during evaluation so success rate measures
    # true tracking stability under initial state perturbations
    eval_cfg.events.pop("push_robot", None)
    eval_motion_cfg = eval_cfg.commands["motion"]
    assert isinstance(eval_motion_cfg, MotionCommandCfg)
    eval_motion_cfg.motion_file = str(motion_path)
    eval_motion_cfg.sampling_mode = "start"

    eval_env = ManagerBasedRlEnv(cfg=eval_cfg, device=device, render_mode=None)
    eval_vec_env = RslRlVecEnvWrapper(eval_env)

    eval_runner = MotionTrackingOnPolicyRunner(
        eval_vec_env, asdict(agent_cfg), log_dir=None, device=device
    )
    eval_runner.load(str(ckpt_path), map_location=device)
    eval_policy = eval_runner.get_inference_policy(device=device)

    obs, _ = eval_vec_env.reset()
    eval_cmd: MotionCommand = eval_env.command_manager.get_term("motion")
    eval_cmd.time_steps.zero_()
    eval_cmd._resample_command(torch.arange(args.num_eval_envs, device=device))
    obs = eval_vec_env.get_observations()

    failed_mask = torch.zeros(args.num_eval_envs, dtype=torch.bool, device=device)
    batch_abs_q_err = []
    batch_sq_q_err = []

    with torch.no_grad():
        for step in range(traj_steps - 1):
            q_diff = eval_cmd.robot.data.joint_pos - eval_cmd.joint_pos
            batch_abs_q_err.append(torch.abs(q_diff).mean().item())
            batch_sq_q_err.append((q_diff ** 2).mean().item())

            actions = eval_policy(obs)
            obs, _, dones, extras = eval_vec_env.step(actions)
            # An episode that terminates before the final step of the motion is a failure
            if step < traj_steps - 2:
                time_outs = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
                terminated = dones.bool() & (~time_outs.bool())
                failed_mask |= terminated

    eval_vec_env.close()

    num_completed = int((~failed_mask).sum().item())
    success_rate = 100.0 * num_completed / args.num_eval_envs

    # Compute comprehensive metrics
    q_err = sim_q_arr - ref_q_arr
    dq_err = sim_dq_arr - ref_dq_arr

    joint_mae_rad = float(np.mean(np.abs(q_err)))
    joint_rmse_rad = float(np.sqrt(np.mean(q_err ** 2)))
    joint_mae_deg = float(np.degrees(joint_mae_rad))
    joint_rmse_deg = float(np.degrees(joint_rmse_rad))

    joint_vel_mae = float(np.mean(np.abs(dq_err)))
    joint_vel_rmse = float(np.sqrt(np.mean(dq_err ** 2)))

    per_joint_mae_rad = np.mean(np.abs(q_err), axis=0)
    per_joint_rmse_rad = np.sqrt(np.mean(q_err ** 2, axis=0))

    mpkpe_m = float(np.mean(mpkpe_list))
    rmpkpe_m = float(np.mean(rmpkpe_list))
    hand_ee_m = float(np.mean(hand_err_list))
    foot_ee_m = float(np.mean(foot_err_list))
    anchor_ori_rad = float(np.mean(anchor_ori_err_list))

    metrics = {
        "robot_model": robot_model_str,
        "task_id": task_id_str,
        "motion_file": str(motion_path.relative_to(WORKSPACE_ROOT)),
        "checkpoint_file": str(ckpt_path.relative_to(WORKSPACE_ROOT)),
        "trajectory_frames_50fps": int(traj_steps),
        "trajectory_duration_sec": float(traj_steps * control_dt),
        "joint_tracking_mae_rad": round(joint_mae_rad, 5),
        "joint_tracking_mae_deg": round(joint_mae_deg, 3),
        "joint_tracking_rmse_rad": round(joint_rmse_rad, 5),
        "joint_tracking_rmse_deg": round(joint_rmse_deg, 3),
        "perturbed_batch_joint_mae_rad": round(float(np.mean(batch_abs_q_err)), 5),
        "perturbed_batch_joint_rmse_rad": round(float(np.sqrt(np.mean(batch_sq_q_err))), 5),
        "joint_velocity_mae_rad_s": round(joint_vel_mae, 4),
        "joint_velocity_rmse_rad_s": round(joint_vel_rmse, 4),
        "mpkpe_m": round(mpkpe_m, 5),
        "mpkpe_mm": round(mpkpe_m * 1000.0, 2),
        "r_mpkpe_m": round(rmpkpe_m, 5),
        "r_mpkpe_mm": round(rmpkpe_m * 1000.0, 2),
        "end_effector_hand_error_m": round(hand_ee_m, 5),
        "end_effector_hand_error_mm": round(hand_ee_m * 1000.0, 2),
        "end_effector_foot_error_m": round(foot_ee_m, 5),
        "end_effector_foot_error_mm": round(foot_ee_m * 1000.0, 2),
        "anchor_orientation_error_rad": round(anchor_ori_rad, 5),
        "anchor_orientation_error_deg": round(float(np.degrees(anchor_ori_rad)), 3),
        "num_eval_episodes": int(args.num_eval_envs),
        "num_completed_episodes": num_completed,
        "success_rate_percent": round(success_rate, 2),
        "per_joint_metrics": {
            name: {
                "mae_rad": round(float(per_joint_mae_rad[idx]), 5),
                "mae_deg": round(float(np.degrees(per_joint_mae_rad[idx])), 3),
                "rmse_rad": round(float(per_joint_rmse_rad[idx]), 5),
                "rmse_deg": round(float(np.degrees(per_joint_rmse_rad[idx])), 3),
            }
            for idx, name in enumerate(joint_names_list)
        },
    }

    metrics_path = (WORKSPACE_ROOT / args.output_metrics).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Print rich summary table
    table = Table(title=f"{args.robot} BeyondMimic Motion Tracking Evaluation Summary")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold green")
    table.add_row("Joint Tracking MAE", f"{joint_mae_rad:.4f} rad ({joint_mae_deg:.2f}°)")
    table.add_row("Joint Tracking RMSE", f"{joint_rmse_rad:.4f} rad ({joint_rmse_deg:.2f}°)")
    table.add_row("Joint Velocity RMSE", f"{joint_vel_rmse:.4f} rad/s")
    table.add_row("World Keybody Error (MPKPE)", f"{mpkpe_m:.4f} m ({mpkpe_m * 1000:.1f} mm)")
    table.add_row("Anchor-Relative Keybody Error (R-MPKPE)", f"{rmpkpe_m:.4f} m ({rmpkpe_m * 1000:.1f} mm)")
    table.add_row("Hand End-Effector Error (Relative)", f"{hand_ee_m:.4f} m ({hand_ee_m * 1000:.1f} mm)")
    table.add_row("Foot End-Effector Error (Relative)", f"{foot_ee_m:.4f} m ({foot_ee_m * 1000:.1f} mm)")
    table.add_row("Torso Anchor Orientation Error", f"{anchor_ori_rad:.4f} rad ({np.degrees(anchor_ori_rad):.2f}°)")
    table.add_row(
        "Motion Completion Success Rate",
        f"{success_rate:.2f}% ({num_completed}/{args.num_eval_envs} episodes)",
    )
    print(table)
    print(f"[bold green]Saved evaluation metrics JSON to {metrics_path}[/bold green]")


if __name__ == "__main__":
    main()

