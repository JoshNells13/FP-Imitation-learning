"""Generates a biomechanically balanced human martial-arts front-kick BVH motion clip (data/bvh/human_kick_xsens.bvh).

Uses the authentic Xsens 23-joint human skeleton and fighting guard posture from
251021_04_boxing_120Hz_cm_3DsMax.bvh, modulating the pelvis weight shift over the
support foot, hip/knee/ankle chamber-extension-recoil kick phases, and upper-body
counter-balance with C2-continuous minimum-jerk trajectories.
"""

from __future__ import annotations

import math
import pathlib
import numpy as np

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent


def min_jerk(u: float) -> float:
    """C2-continuous minimum-jerk interpolation on [0, 1]."""
    u = max(0.0, min(1.0, u))
    return 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)


def bell_curve(t: float, t_start: float, t_peak1: float, t_peak2: float, t_end: float) -> float:
    """Smooth C2 plateau envelope rising on [t_start, t_peak1], holding on [t_peak1, t_peak2], falling on [t_peak2, t_end]."""
    if t <= t_start or t >= t_end:
        return 0.0
    if t < t_peak1:
        return min_jerk((t - t_start) / (t_peak1 - t_start))
    if t <= t_peak2:
        return 1.0
    return 1.0 - min_jerk((t - t_peak2) / (t_end - t_peak2))


def generate_kick_bvh(
    source_bvh: pathlib.Path,
    output_bvh: pathlib.Path,
    num_frames: int = 360,
    fps: int = 120,
) -> None:
    lines = source_bvh.read_text(encoding="utf-8").splitlines()
    motion_idx = lines.index("MOTION")
    header_lines = lines[: motion_idx + 1]
    base_frame_lines = lines[motion_idx + 3 : motion_idx + 3 + num_frames]

    motion_data = np.array(
        [[float(x) for x in line.strip().split()] for line in base_frame_lines],
        dtype=np.float64,
    )

    # Use a steady guard posture from frame 0 as the baseline and add subtle natural breathing/guard motion
    base_pose = motion_data[0].copy()
    out_data = np.tile(base_pose, (num_frames, 1))

    # Channel indices in Xsens 3dsMax BVH (72 channels total):
    # 0..2: Hips Xpos, Ypos, Zpos (cm)
    # 3..5: Hips Yrot, Xrot, Zrot (deg)
    # 6..8: Chest Yrot, Xrot, Zrot
    # 9..11: Chest2 Yrot, Xrot, Zrot
    # 27..29: RightShoulder Yrot, Xrot, Zrot
    # 30..32: RightElbow Yrot, Xrot, Zrot
    # 39..41: LeftShoulder Yrot, Xrot, Zrot
    # 42..44: LeftElbow Yrot, Xrot, Zrot
    # 48..50: RightHip Yrot, Xrot, Zrot
    # 51..53: RightKnee Yrot, Xrot, Zrot
    # 54..56: RightAnkle Yrot, Xrot, Zrot
    # 60..62: LeftHip Yrot, Xrot, Zrot
    # 63..65: LeftKnee Yrot, Xrot, Zrot
    # 66..68: LeftAnkle Yrot, Xrot, Zrot

    for f in range(num_frames):
        t = f / float(fps)  # 0.0 to 3.0 seconds

        # 1. Weight shift envelope onto Left support leg (0.3s -> 0.8s, held until 2.2s, return 2.7s)
        w_shift = bell_curve(t, 0.25, 0.75, 2.15, 2.70)

        # 2. Right Hip flexion envelope (lifts thigh up for chamber + kick + rechamber)
        hip_lift = bell_curve(t, 0.50, 1.05, 1.85, 2.45)
        # Extra peak hip lift during full kick extension (1.10s -> 1.75s)
        kick_ext = bell_curve(t, 1.00, 1.30, 1.60, 1.95)

        # 3. Right Knee bend envelope:
        # Chambers during [0.50, 1.05], extends straight during [1.05, 1.60], re-chambers [1.60, 2.00], lands [2.45]
        chamber_env = bell_curve(t, 0.45, 0.95, 2.00, 2.45)
        knee_bend_deg = 78.0 * chamber_env * (1.0 - 0.88 * kick_ext)

        # Apply Hips weight shift over left support leg & slight support knee bend
        # Since Hips Yrotation ~ 178.6 deg, world +X corresponds to -Xposition
        out_data[f, 0] = base_pose[0] - 7.5 * w_shift
        out_data[f, 1] = base_pose[1] - 1.5 * w_shift
        out_data[f, 2] = base_pose[2] - 2.0 * kick_ext

        # Support leg (LeftHip, LeftKnee, LeftAnkle) stabilization
        out_data[f, 61] = base_pose[61] - 8.0 * w_shift   # LeftHip Xrot slight flexion
        out_data[f, 64] = base_pose[64] + 14.0 * w_shift  # LeftKnee Xrot athletic bend
        out_data[f, 67] = base_pose[67] - 6.0 * w_shift   # LeftAnkle Xrot dorsiflexion

        # Kicking leg (RightHip, RightKnee, RightAnkle)
        # Negative Xrot on RightHip lifts the right thigh forward & upward
        out_data[f, 49] = base_pose[49] - 68.0 * hip_lift - 18.0 * kick_ext
        # Positive Xrot on RightKnee bends the right knee backward; near-zero extends the kick
        out_data[f, 52] = base_pose[52] + knee_bend_deg
        # RightAnkle dorsiflexion / kick strike alignment
        out_data[f, 55] = base_pose[55] - 15.0 * hip_lift + 25.0 * kick_ext

        # Counter-balance torso & fighting guard arms
        out_data[f, 4] = base_pose[4] + 5.0 * hip_lift + 6.0 * kick_ext   # Hips Xrot lean back slightly
        out_data[f, 5] = base_pose[5] - 4.0 * w_shift                     # Hips Zrot lateral balance
        out_data[f, 7] = base_pose[7] + 4.0 * kick_ext                    # Chest Xrot counter-balance
        out_data[f, 10] = base_pose[10] + 4.0 * kick_ext                  # Chest2 Xrot

        # Guard arms: bring elbows and shoulders into compact martial arts guard
        # Note: in Xsens 3dsMax BVH, zero pose already has arms down at the sides.
        # Negative Xrotation (indices 28, 31 for Right; 40, 43 for Left) pitches arms & forearms forward into guard.
        guard_env = bell_curve(t, 0.15, 0.60, 2.40, 2.85)
        out_data[f, 28] = base_pose[28] - 18.0 * guard_env + 12.0 * kick_ext  # RightShoulder Xrot (forward guard + kick counter-swing)
        out_data[f, 29] = base_pose[29] - 4.0 * guard_env                     # RightShoulder Zrot
        out_data[f, 31] = base_pose[31] - 48.0 * guard_env                    # RightElbow Xrot (bend forearm up into guard)

        out_data[f, 40] = base_pose[40] - 24.0 * guard_env - 10.0 * kick_ext  # LeftShoulder Xrot (forward guard)
        out_data[f, 41] = base_pose[41] + 4.0 * guard_env                     # LeftShoulder Zrot
        out_data[f, 43] = base_pose[43] - 52.0 * guard_env                    # LeftElbow Xrot (bend forearm up into guard)

    frame_strings = [" ".join(f"{v:.6f}" for v in row) for row in out_data]
    bvh_text = (
        "\n".join(header_lines)
        + f"\nFrames: {num_frames}\n"
        + f"Frame Time: {1.0 / fps:.6f}\n"
        + "\n".join(frame_strings)
        + "\n"
    )
    output_bvh.parent.mkdir(parents=True, exist_ok=True)
    output_bvh.write_text(bvh_text, encoding="utf-8")
    print(f"Generated human kick BVH at {output_bvh} ({num_frames} frames @ {fps} Hz)")


if __name__ == "__main__":
    src = (
        WORKSPACE_ROOT
        / "third_party"
        / "GMR"
        / "assets"
        / "xsens_bvh_test"
        / "251021_04_boxing_120Hz_cm_3DsMax.bvh"
    )
    dst = WORKSPACE_ROOT / "data" / "bvh" / "human_kick_xsens.bvh"
    generate_kick_bvh(src, dst)
