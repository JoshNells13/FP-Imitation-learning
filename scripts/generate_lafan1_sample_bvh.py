"""Generates a clean LAFAN1-hierarchy human motion capture BVH clip for testing bvh_lafan1 retargeting."""

from __future__ import annotations

import math
import pathlib
import numpy as np

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent


LAFAN1_BVH_HEADER = """HIERARCHY
ROOT Hips
{
    OFFSET 0.00000 0.00000 0.00000
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT LeftUpLeg
    {
        OFFSET 0.00000 10.50000 0.00000
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT LeftLeg
        {
            OFFSET 43.00000 0.00000 0.00000
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT LeftFoot
            {
                OFFSET 42.00000 0.00000 0.00000
                CHANNELS 3 Zrotation Yrotation Xrotation
                JOINT LeftToe
                {
                    OFFSET 0.00000 0.00000 15.00000
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    End Site
                    {
                        OFFSET 0.00000 0.00000 5.00000
                    }
                }
            }
        }
    }
    JOINT RightUpLeg
    {
        OFFSET 0.00000 -10.50000 0.00000
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT RightLeg
        {
            OFFSET 43.00000 0.00000 0.00000
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT RightFoot
            {
                OFFSET 42.00000 0.00000 0.00000
                CHANNELS 3 Zrotation Yrotation Xrotation
                JOINT RightToe
                {
                    OFFSET 0.00000 0.00000 15.00000
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    End Site
                    {
                        OFFSET 0.00000 0.00000 5.00000
                    }
                }
            }
        }
    }
    JOINT Spine
    {
        OFFSET -12.00000 0.00000 0.00000
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT Spine1
        {
            OFFSET -15.00000 0.00000 0.00000
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT Spine2
            {
                OFFSET -15.00000 0.00000 0.00000
                CHANNELS 3 Zrotation Yrotation Xrotation
                JOINT Neck
                {
                    OFFSET -14.00000 0.00000 0.00000
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    JOINT Head
                    {
                        OFFSET -12.00000 0.00000 0.00000
                        CHANNELS 3 Zrotation Yrotation Xrotation
                        End Site
                        {
                            OFFSET -10.00000 0.00000 0.00000
                        }
                    }
                }
                JOINT LeftShoulder
                {
                    OFFSET -10.00000 8.00000 0.00000
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    JOINT LeftArm
                    {
                        OFFSET 0.00000 12.00000 0.00000
                        CHANNELS 3 Zrotation Yrotation Xrotation
                        JOINT LeftForeArm
                        {
                            OFFSET 0.00000 28.00000 0.00000
                            CHANNELS 3 Zrotation Yrotation Xrotation
                            JOINT LeftHand
                            {
                                OFFSET 0.00000 25.00000 0.00000
                                CHANNELS 3 Zrotation Yrotation Xrotation
                                End Site
                                {
                                    OFFSET 0.00000 10.00000 0.00000
                                }
                            }
                        }
                    }
                }
                JOINT RightShoulder
                {
                    OFFSET -10.00000 -8.00000 0.00000
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    JOINT RightArm
                    {
                        OFFSET 0.00000 -12.00000 0.00000
                        CHANNELS 3 Zrotation Yrotation Xrotation
                        JOINT RightForeArm
                        {
                            OFFSET 0.00000 -28.00000 0.00000
                            CHANNELS 3 Zrotation Yrotation Xrotation
                            JOINT RightHand
                            {
                                OFFSET 0.00000 -25.00000 0.00000
                                CHANNELS 3 Zrotation Yrotation Xrotation
                                End Site
                                {
                                    OFFSET 0.00000 -10.00000 0.00000
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
"""


def generate_lafan1_bvh(output_path: pathlib.Path, num_frames: int = 90, fps: int = 30) -> None:
    """Write a 22-joint LAFAN1 BVH file with a balanced upper-body gesture & knee bend."""
    num_joints = 22
    frames = []
    for f in range(num_frames):
        t = f / float(fps)
        phase = 2.0 * math.pi * t / 3.0
        row = np.zeros(3 + 3 * num_joints, dtype=np.float64)
        # In LAFAN1 raw coordinates (before Z-up rotation [[0,0,1],[1,0,0],[0,1,0]]):
        # Raw Y becomes Z (vertical height in cm -> divided by 100 in extract.py)
        # Note: with Hips Zrotation=90, local +X points down (-Y raw)
        row[0] = 0.0
        row[1] = 88.0 - 4.0 * math.sin(phase) ** 2
        row[2] = 0.0
        # Hips ZYX Euler (Z=90 aligns local -X spine upward along raw +Y -> world +Z)
        row[3] = 90.0
        row[4] = 0.0
        row[5] = 0.0
        frames.append(" ".join(f"{val:.5f}" for val in row))

    content = (
        LAFAN1_BVH_HEADER
        + "MOTION\n"
        + f"Frames: {num_frames}\n"
        + f"Frame Time: {1.0 / fps:.6f}\n"
        + "\n".join(frames)
        + "\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    out = WORKSPACE_ROOT / "data" / "bvh" / "lafan1_sample.bvh"
    generate_lafan1_bvh(out)
    print(f"Generated LAFAN1 sample BVH at {out}")

