"""BeyondMimic PPO training script for Booster K1 whole-body motion tracking in MJLab.

Trains a 22-DoF Booster K1 humanoid policy in MuJoCo + MJLab
(Mjlab-Tracking-Flat-Booster-K1) using the BeyondMimic methodology:
  - Anchor-yaw-aligned relative body pose tracking (Trunk anchor)
  - 14 tracked keybodies across upper and lower body
  - 125-dim actor observation / 251-dim privileged critic observation
  - 6 exponential tracking reward terms + 3 smoothness/limit regularizers
  - Adaptive failure-bin starting frame sampling + Reference State Initialization
  - Task-space height & orientation termination criteria
  - PPO via rsl-rl-lib with [512, 256, 128] ELU MLP actor-critic and observation normalization
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import traceback
from dataclasses import asdict
from datetime import datetime

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
import yaml
from rich import print

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.config.k1.env_cfgs import booster_k1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.k1.rl_cfg import booster_k1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


class SafeMotionTrackingRunner(MotionTrackingOnPolicyRunner):
    """MotionTrackingOnPolicyRunner that safely handles checkpoint + ONNX export on Windows."""

    def save(self, path: str, infos=None) -> None:
        # Always save the PyTorch checkpoint first via MjlabOnPolicyRunner.save
        super(MotionTrackingOnPolicyRunner, self).save(path, infos)
        try:
            from mjlab.tasks.tracking.rl.exporter import (
                attach_onnx_metadata,
                export_motion_policy_as_onnx,
            )

            policy_dir = str(pathlib.Path(path).parent) + os.sep
            filename = pathlib.Path(path).parent.name + ".onnx"
            normalizer = (
                self.alg.policy.actor_obs_normalizer
                if self.alg.policy.actor_obs_normalization
                else None
            )
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                self.alg.policy,
                normalizer=normalizer,
                path=policy_dir,
                filename=filename,
            )
            attach_onnx_metadata(
                self.env.unwrapped,
                "local_tensorboard_run",
                path=policy_dir,
                filename=filename,
            )
        except Exception as exc:
            print(f"[yellow]Warning: ONNX export skipped for {path}: {exc}[/yellow]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train BeyondMimic PPO policy for Booster K1 (22-DoF) in MJLab."
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="booster_k1",
        choices=["booster_k1"],
        help="Target humanoid robot model (booster_k1).",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default="data/motions/k1_kick.npz",
        help="Path to local MJLab .npz reference motion file.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=2048,
        help="Number of parallel MuJoCo-Warp environments (default: 2048).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="Number of PPO training iterations (default: 500).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Checkpoint save interval in iterations (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="beyondmimic_k1_kick",
        help="Run name tag for logs/rsl_rl.",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=str,
        default="outputs/checkpoints/k1_kick_policy.pt",
        help="Path to copy the final trained checkpoint.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    args = parser.parse_args()

    motion_path = pathlib.Path(args.motion_file).resolve()
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion NPZ file not found: {motion_path}")

    os.environ["MUJOCO_GL"] = "egl"
    selected_gpus, num_gpus = select_gpus([0])
    device = "cpu" if num_gpus == 0 else f"cuda:{selected_gpus[0]}"
    configure_torch_backends()

    print(f"[bold blue]Initializing BeyondMimic PPO Training ({args.robot}) on {device}[/bold blue]")
    print(f"  Motion file:    {motion_path}")
    print(f"  Num envs:       {args.num_envs}")
    print(f"  Max iterations: {args.max_iterations}")

    env_cfg = booster_k1_flat_tracking_env_cfg(play=False)
    agent_cfg = booster_k1_tracking_ppo_runner_cfg()

    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.save_interval = args.save_interval
    agent_cfg.logger = "tensorboard"
    agent_cfg.run_name = args.run_name

    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)

    log_root_path = (WORKSPACE_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name).resolve()
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = log_root_path / f"{time_str}_{agent_cfg.run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = SafeMotionTrackingRunner(
        env, asdict(agent_cfg), str(log_dir), device=device
    )

    if args.resume_from is not None:
        resume_path = pathlib.Path(args.resume_from).resolve()
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()

    # Copy final checkpoint to deterministic path
    latest_ckpts = sorted(
        log_dir.glob("model_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if latest_ckpts:
        final_ckpt = latest_ckpts[-1]
        out_ckpt_path = (WORKSPACE_ROOT / args.output_checkpoint).resolve()
        out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_ckpt, out_ckpt_path)
        print(f"[bold green]Copied final trained policy checkpoint to {out_ckpt_path}[/bold green]")

        onnx_files = list(log_dir.glob("*.onnx"))
        if onnx_files:
            out_onnx_path = out_ckpt_path.with_suffix(".onnx")
            shutil.copy2(onnx_files[-1], out_onnx_path)
            print(f"[green]Copied exported ONNX policy to {out_onnx_path}[/green]")


if __name__ == "__main__":
    main()

