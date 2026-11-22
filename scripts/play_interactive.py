"""Interactive 3D viewer script for Booster K1 (22-DoF) in MJLab.

Supports:
  - Native MuJoCo desktop GLFW viewer (--viewer native)
  - Browser-based Viser 3D web viewer (--viewer viser)
  - Reference-only playback (--reference-only) or trained policy inference (--checkpoint-file)
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
import torch
from rich import print

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.config.k1.env_cfgs import booster_k1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.k1.rl_cfg import booster_k1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch interactive 3D viewer for Booster K1 (22-DoF) motion tracking."
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="booster_k1",
        choices=["booster_k1"],
        help="Robot model to display (booster_k1).",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default="data/motions/k1_kick.npz",
        help="Path to MJLab .npz reference motion file.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Optional path to trained .pt model checkpoint.",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Play the reference trajectory kinematically without requiring a trained policy.",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        choices=["native", "viser"],
        default="native",
        help="3D viewer backend: 'native' (MuJoCo window) or 'viser' (browser at http://localhost:8080).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of robots to display simultaneously (default: 1).",
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    motion_path = (WORKSPACE_ROOT / args.motion_file).resolve()

    if not motion_path.exists():
        raise FileNotFoundError(f"Motion NPZ not found: {motion_path}")

    env_cfg = booster_k1_flat_tracking_env_cfg(play=True)
    agent_cfg = booster_k1_tracking_ppo_runner_cfg()

    env_cfg.scene.num_envs = args.num_envs
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    vec_env = RslRlVecEnvWrapper(env)

    ckpt_path = (WORKSPACE_ROOT / args.checkpoint_file).resolve() if args.checkpoint_file else None

    if args.reference_only or ckpt_path is None or not ckpt_path.exists():
        print(f"[cyan]Running in reference-motion preview mode for {args.robot} ({motion_path.name})[/cyan]")
        cmd_term = env.command_manager.get_term("motion")
        action_dim = env.action_manager.total_action_dim

        def ref_policy(obs):
            # Step reference state kinematically onto the robot for clean preview
            cmd_term.robot.write_root_state_to_sim(
                torch.cat(
                    [
                        cmd_term.body_pos_w[:, 0],
                        cmd_term.body_quat_w[:, 0],
                        cmd_term.body_lin_vel_w[:, 0],
                        cmd_term.body_ang_vel_w[:, 0],
                    ],
                    dim=-1,
                )
            )
            cmd_term.robot.write_joint_state_to_sim(cmd_term.joint_pos, cmd_term.joint_vel)
            return torch.zeros((env.num_envs, action_dim), device=device)

        policy = ref_policy
    else:
        print(f"[bold green]Loading trained checkpoint:[/bold green] {ckpt_path}")
        runner = MotionTrackingOnPolicyRunner(
            vec_env, asdict(agent_cfg), log_dir=None, device=device
        )
        runner.load(str(ckpt_path), map_location=device)
        policy = runner.get_inference_policy(device=device)

    if args.viewer == "native":
        NativeMujocoViewer(vec_env, policy).run()
    else:
        ViserPlayViewer(vec_env, policy).run()

    vec_env.close()


if __name__ == "__main__":
    main()