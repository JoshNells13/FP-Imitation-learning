"""End-to-end orchestrator script for the GMR -> MJLab -> BeyondMimic PPO -> Evaluation pipeline."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from rich import print

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYTHON_EXE = sys.executable


def run_step(title: str, cmd: list[str]) -> None:
    print(f"\n[bold cyan]=== {title} ===[/bold cyan]")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete GMR + BeyondMimic + MJLab humanoid motion imitation pipeline."
    )
    parser.add_argument(
        "--bvh-file",
        type=str,
        default="data/bvh/human_kick_xsens.bvh",
        help="Input human .bvh file.",
    )
    parser.add_argument("--num-envs", type=int, default=2048, help="Parallel environments for PPO.")
    parser.add_argument("--max-iterations", type=int, default=500, help="PPO training iterations.")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip PPO training and run evaluation using the existing checkpoint.",
    )
    args = parser.parse_args()

    run_step(
        "Stage 1: GMR Human-to-Booster-K1 Motion Retargeting",
        [
            PYTHON_EXE,
            "scripts/retarget_bvh_gmr.py",
            "--robot",
            "booster_k1",
            "--bvh-file",
            args.bvh_file,
            "--reset-to-zero",
            "--save-pkl",
            "data/retargeted/k1_kick.pkl",
            "--save-csv",
            "data/retargeted/k1_kick.csv",
            "--save-video",
            "outputs/videos/gmr_retargeted_k1_kick.mp4",
        ],
    )

    run_step(
        "Stage 2: Convert Retargeted CSV to MJLab 50 FPS NPZ Reference",
        [
            PYTHON_EXE,
            "scripts/convert_csv_to_npz.py",
            "--robot",
            "booster_k1",
            "--input-csv",
            "data/retargeted/k1_kick.csv",
            "--output-npz",
            "data/motions/k1_kick.npz",
            "--output-video",
            "outputs/videos/k1_kick_reference.mp4",
        ],
    )

    if not args.skip_training:
        run_step(
            "Stage 3: BeyondMimic PPO Whole-Body Tracking Training (Booster K1)",
            [
                PYTHON_EXE,
                "scripts/train_ppo.py",
                "--robot",
                "booster_k1",
                "--motion-file",
                "data/motions/k1_kick.npz",
                "--num-envs",
                str(args.num_envs),
                "--max-iterations",
                str(args.max_iterations),
                "--save-interval",
                "100",
                "--run-name",
                "beyondmimic_k1_kick",
                "--output-checkpoint",
                "outputs/checkpoints/k1_kick_policy.pt",
            ],
        )

    run_step(
        "Stage 4: Policy Evaluation, Metrics Calculation & Video Recording",
        [
            PYTHON_EXE,
            "scripts/evaluate_and_play.py",
            "--robot",
            "booster_k1",
            "--checkpoint-file",
            "outputs/checkpoints/k1_kick_policy.pt",
            "--motion-file",
            "data/motions/k1_kick.npz",
            "--output-video",
            "outputs/videos/k1_trained_policy_tracking.mp4",
            "--output-metrics",
            "outputs/evaluation/k1_metrics.json",
            "--output-plot",
            "outputs/evaluation/k1_joint_tracking_comparison.png",
        ],
    )

    run_step(
        "Stage 5: Generate Side-by-Side Comparison Video",
        [PYTHON_EXE, "scripts/create_comparison_video.py"],
    )


if __name__ == "__main__":
    main()

