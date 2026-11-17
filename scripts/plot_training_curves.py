"""Extracts TensorBoard scalars from MJLab BeyondMimic PPO training and plots convergence curves."""

from __future__ import annotations

import os
import pathlib
import sys

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = WORKSPACE_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich import print
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_scalars(event_dir: pathlib.Path) -> dict[str, dict[str, list[float]]]:
    acc = EventAccumulator(str(event_dir))
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    data: dict[str, dict[str, list[float]]] = {}
    for tag in tags:
        events = acc.Scalars(tag)
        data[tag] = {
            "steps": [int(e.step) for e in events],
            "values": [float(e.value) for e in events],
        }
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BeyondMimic PPO training curves from TensorBoard logs.")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Specific run log directory (defaults to latest under logs/rsl_rl/k1_tracking).",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="outputs/evaluation/ppo_training_curves.png",
        help="Output .png path for training convergence curves.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/evaluation/training_summary.json",
        help="Output .json path for training scalar summary.",
    )
    args = parser.parse_args()

    if args.log_dir is not None:
        run_dir = pathlib.Path(args.log_dir).resolve()
    else:
        exp_root = WORKSPACE_ROOT / "logs" / "rsl_rl" / "k1_tracking"
        runs = sorted([d for d in exp_root.iterdir() if d.is_dir()])
        if not runs:
            raise FileNotFoundError(f"No runs found in {exp_root}")
        run_dir = runs[-1]

    print(f"[bold blue]Extracting TensorBoard scalars from:[/bold blue] {run_dir}")
    scalars = extract_scalars(run_dir)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: Mean Reward
    ax = axes[0, 0]
    if "Train/mean_reward" in scalars:
        s = scalars["Train/mean_reward"]
        ax.plot(s["steps"], s["values"], "C0-", linewidth=2.0, label="Mean Episode Reward")
    ax.set_title("Mean Episode Return", fontsize=11, fontweight="bold")
    ax.set_xlabel("PPO Iteration")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # Panel 2: Mean Episode Length
    ax = axes[0, 1]
    if "Train/mean_episode_length" in scalars:
        s = scalars["Train/mean_episode_length"]
        ax.plot(s["steps"], s["values"], "C2-", linewidth=2.0, label="Mean Episode Length (steps)")
    ax.set_title("Episode Horizon Survival (steps)", fontsize=11, fontweight="bold")
    ax.set_xlabel("PPO Iteration")
    ax.set_ylabel("Steps")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # Panel 3: Tracking Errors (Body & Joint)
    ax = axes[1, 0]
    for tag, label, color in [
        ("Metrics/motion/error_joint_pos", "Joint Position Error (rad)", "C1"),
        ("Metrics/motion/error_body_pos", "Relative Body Pos Error (m)", "C3"),
        ("Metrics/motion/error_anchor_pos", "Anchor Pos Error (m)", "C4"),
    ]:
        if tag in scalars:
            s = scalars[tag]
            ax.plot(s["steps"], s["values"], color=color, linewidth=1.8, label=label)
    ax.set_title("Training Tracking Error Metrics", fontsize=11, fontweight="bold")
    ax.set_xlabel("PPO Iteration")
    ax.set_ylabel("Error")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8)

    # Panel 4: Key Tracking Rewards
    ax = axes[1, 1]
    for tag, label in [
        ("Episode_Reward/motion_body_pos", "Body Pos Reward"),
        ("Episode_Reward/motion_body_ori", "Body Ori Reward"),
        ("Episode_Reward/motion_global_root_pos", "Anchor Pos Reward"),
        ("Episode_Reward/motion_global_root_ori", "Anchor Ori Reward"),
    ]:
        if tag in scalars:
            s = scalars[tag]
            ax.plot(s["steps"], s["values"], linewidth=1.6, label=label)
    ax.set_title("BeyondMimic Exponential Tracking Rewards", fontsize=11, fontweight="bold")
    ax.set_xlabel("PPO Iteration")
    ax.set_ylabel("Episode Reward Component")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"BeyondMimic PPO Training Convergence — Booster K1 ({run_dir.name})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = (WORKSPACE_ROOT / args.output_plot).resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"[green]Saved PPO training curves plot to {plot_path}[/green]")

    summary = {
        "run_dir": str(run_dir.relative_to(WORKSPACE_ROOT)),
        "final_scalars": {
            tag: {
                "initial_step": val["steps"][0],
                "initial_value": round(val["values"][0], 5),
                "final_step": val["steps"][-1],
                "final_value": round(val["values"][-1], 5),
            }
            for tag, val in scalars.items()
            if val["values"]
        },
    }
    json_path = (WORKSPACE_ROOT / args.output_json).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[green]Saved training scalar summary to {json_path}[/green]")


if __name__ == "__main__":
    main()

