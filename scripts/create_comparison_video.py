"""Creates a side-by-side synchronized MP4 comparison of the GMR/MJLab reference motion and the trained BeyondMimic policy."""

from __future__ import annotations

import pathlib
import cv2
import imageio
import numpy as np
from rich import print

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent


def add_banner(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 58), (20, 24, 32), -1)
    cv2.putText(
        out,
        title,
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        subtitle,
        (16, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def main() -> None:
    ref_path = WORKSPACE_ROOT / "outputs" / "videos" / "k1_kick_reference.mp4"
    pol_path = WORKSPACE_ROOT / "outputs" / "videos" / "k1_trained_policy_tracking.mp4"
    out_path = WORKSPACE_ROOT / "outputs" / "videos" / "k1_side_by_side_comparison.mp4"

    ref_reader = imageio.get_reader(str(ref_path))
    pol_reader = imageio.get_reader(str(pol_path))

    ref_frames = [f for f in ref_reader]
    pol_frames = [f for f in pol_reader]
    ref_reader.close()
    pol_reader.close()

    # Loop reference frames if policy video has 2 loops
    if len(pol_frames) > len(ref_frames):
        repeats = (len(pol_frames) // len(ref_frames)) + 1
        ref_frames = (ref_frames * repeats)[: len(pol_frames)]

    n_frames = min(len(ref_frames), len(pol_frames))
    combined_frames = []
    for idx in range(n_frames):
        left = add_banner(
            ref_frames[idx],
            "Stage 1-2: GMR Retargeted Reference (Kinematic)",
            "Booster K1 (22-DoF) | 50 Hz Kick Reference Trajectory",
        )
        right = add_banner(
            pol_frames[idx],
            "Stage 3-4: BeyondMimic PPO Policy (Dynamic MuJoCo)",
            "Booster K1 (22-DoF) | Closed-Loop PPO Policy Tracking",
        )
        divider = np.full((left.shape[0], 4, 3), 240, dtype=np.uint8)
        combined = np.hstack([left, divider, right])
        combined_frames.append(combined)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), combined_frames, fps=50, macro_block_size=1)
    print(f"[bold green]Saved side-by-side comparison video to {out_path} ({n_frames} frames @ 50 FPS)[/bold green]")


if __name__ == "__main__":
    main()

