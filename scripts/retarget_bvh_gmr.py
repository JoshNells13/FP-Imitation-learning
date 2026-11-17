"""Headless GMR (General Motion Retargeting) BVH-to-Booster-K1 pipeline.

Retargets human motion capture data (.bvh in either Xsens or LAFAN1 format)
to the 22-DoF Booster K1 humanoid using GMR's Mink + DAQP configuration-space
differential IK solver.

Outputs:
1. GMR pickle file (.pkl) containing fps, root_pos, root_rot (xyzw), dof_pos.
2. BeyondMimic / MJLab 29-column CSV file (.csv):
   [x, y, z, qx, qy, qz, qw, j_0, ..., j_21]
3. Optional offscreen-rendered MuJoCo preview video (.mp4) of the retargeted motion.
"""

from __future__ import annotations

import os
import pathlib
import pickle
import sys
import types

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = WORKSPACE_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("WP_CACHE_PATH", str(CACHE_DIR / "warp"))

# Ensure local third_party packages are importable
for rel_path in ("third_party/GMR", "third_party/mjlab/src"):
    full_path = str(WORKSPACE_ROOT / rel_path)
    if full_path not in sys.path:
        sys.path.insert(0, full_path)


def _install_headless_curve_editor_stub() -> None:
    """Provide a lightweight headless OffsetManager so Xsens BVH parsing does not require PyQt6."""
    mod_name = "general_motion_retargeting.utils.xsens_vendor.bvh_edit.CurveEditor"
    if mod_name in sys.modules:
        return

    stub = types.ModuleType(mod_name)

    class OffsetManager:
        def __init__(self, default_path: str = "offsets.json", default_Processing_Node: int = 23, **kwargs):
            self.default_path = default_path
            self.node_count = default_Processing_Node

        def load_offsets(self) -> dict:
            return {}

        def parse_to_window_format(self, names: list[str], loaded_offsets: dict) -> dict[tuple[int, int], float]:
            return {
                (node_idx, axis_idx): 0.0
                for node_idx in range(len(names))
                for axis_idx in range(3)
            }

    stub.OffsetManager = OffsetManager  # type: ignore[attr-defined]
    sys.modules[mod_name] = stub


_install_headless_curve_editor_stub()

import argparse
import imageio
import mujoco
import numpy as np
from rich import print
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file as load_lafan1_bvh
from general_motion_retargeting.utils.xsens import load_xsens_file


def detect_bvh_format(bvh_path: pathlib.Path) -> str:
    """Auto-detect whether a BVH file uses LAFAN1 or Xsens joint naming."""
    text = bvh_path.read_text(encoding="utf-8", errors="ignore")[:4096]
    if "LeftUpLeg" in text or "Spine1" in text:
        return "lafan1"
    return "xsens"


def adjust_ground_contact(
    xml_path: str,
    qpos_seq: np.ndarray,
    robot: str = "booster_k1",
) -> np.ndarray:
    """Ensure the robot's lowest foot link height rests cleanly on the ground plane."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    left_name, right_name = "left_foot_link", "right_foot_link"
    target_min_foot_z = 0.024

    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_name)
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_name)

    foot_z_list = []
    for t in range(qpos_seq.shape[0]):
        data.qpos[:] = qpos_seq[t]
        mujoco.mj_forward(model, data)
        lz = data.xpos[left_foot_id, 2]
        rz = data.xpos[right_foot_id, 2]
        foot_z_list.append(min(lz, rz))

    # Use 5th percentile of lowest foot height as ground reference
    min_foot_z = float(np.percentile(foot_z_list, 5))
    z_offset = target_min_foot_z - min_foot_z
    print(
        f"[cyan]Ground alignment ({robot}):[/cyan] 5th-percentile {left_name} z = {min_foot_z:.4f} m, "
        f"applying z_offset = {z_offset:+.4f} m (target = {target_min_foot_z:.4f} m)"
    )
    adjusted = qpos_seq.copy()
    adjusted[:, 2] += z_offset
    return adjusted


def render_preview_video(
    xml_path: str,
    qpos_seq: np.ndarray,
    fps: int,
    video_path: pathlib.Path,
    robot: str = "booster_k1",
    width: int = 960,
    height: int = 720,
) -> None:
    """Render an offscreen MuJoCo preview video of the retargeted humanoid motion."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 1.85
    cam.elevation = -14.0
    cam.azimuth = 135.0
    root_body = "Trunk"
    lookat_z = 0.50

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)

    frames = []
    for t in range(qpos_seq.shape[0]):
        data.qpos[:] = qpos_seq[t]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.xpos[root_id]
        cam.lookat[2] = lookat_z
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    renderer.close()
    imageio.mimsave(str(video_path), frames, fps=fps, macro_block_size=1)
    print(f"[green]Saved retargeted preview video to {video_path}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retarget human BVH motion to Booster K1 (22-DoF) via GMR."
    )
    parser.add_argument(
        "--bvh-file",
        type=str,
        required=True,
        help="Path to input .bvh human motion file.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["auto", "lafan1", "xsens"],
        default="auto",
        help="BVH skeleton format (auto, lafan1, or xsens).",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="booster_k1",
        choices=["booster_k1"],
        help="Target robot model (booster_k1).",
    )
    parser.add_argument(
        "--save-pkl",
        type=str,
        default=None,
        help="Output .pkl path for GMR motion data.",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Output 36-column .csv path for MJLab / BeyondMimic.",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Optional output .mp4 path for offscreen MuJoCo preview.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Optional start frame index in the source BVH.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Optional end frame index in the source BVH.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=30,
        help="Target frame rate for Xsens BVH downsampling (default: 30 FPS).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="Unit scale for Xsens BVH coordinates (0.01 for cm, 1.0 for m).",
    )
    parser.add_argument(
        "--reset-to-zero",
        action="store_true",
        help="Reset initial horizontal XY position and yaw orientation to origin.",
    )
    args = parser.parse_args()

    bvh_path = pathlib.Path(args.bvh_file).resolve()
    if not bvh_path.exists():
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")

    bvh_format = detect_bvh_format(bvh_path) if args.format == "auto" else args.format
    print(f"[bold blue]Loading BVH file:[/bold blue] {bvh_path} (format={bvh_format})")

    if bvh_format == "lafan1":
        lafan1_frames, actual_human_height = load_lafan1_bvh(str(bvh_path))
        if args.start is not None or args.end is not None:
            lafan1_frames = lafan1_frames[args.start : args.end]
        aligned_fps = 30
        src_human = "bvh_lafan1"
    elif bvh_format == "xsens":
        xsens_args = argparse.Namespace(
            bvh_file=str(bvh_path),
            scale=args.scale,
            start=args.start,
            end=args.end,
            bvh_format="3DSM",
            reset_to_zero=args.reset_to_zero,
            target_fps=args.target_fps,
        )
        lafan1_frames, actual_human_height, frame_time = load_xsens_file(xsens_args)
        src_fps = max(1, int(round(1.0 / float(frame_time))))
        stride = max(1, int(round(src_fps / args.target_fps)))
        lafan1_frames = lafan1_frames[::stride]
        aligned_fps = int(round(src_fps / stride))
        src_human = "bvh_xsens"
    else:
        raise ValueError(f"Unsupported format: {bvh_format}")

    print(
        f"[cyan]Loaded {len(lafan1_frames)} frames at {aligned_fps} FPS "
        f"(estimated human height: {actual_human_height:.3f} m)[/cyan]"
    )

    retargeter = GMR(
        src_human=src_human,
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        solver="daqp",
        damping=0.5,
        verbose=False,
    )

    qpos_list = []
    for i in tqdm(range(len(lafan1_frames)), desc="GMR Retargeting"):
        qpos = retargeter.retarget(lafan1_frames[i])
        qpos_list.append(qpos.copy())

    qpos_seq = np.asarray(qpos_list, dtype=np.float64)
    xml_path = str(ROBOT_XML_DICT[args.robot])
    qpos_seq = adjust_ground_contact(xml_path, qpos_seq, robot=args.robot)

    # Extract components:
    # MuJoCo qpos is [x, y, z, qw, qx, qy, qz, j0..j_{N-1}]
    # Both GMR .pkl (bvh_to_robot.py) and BeyondMimic / MJLab .csv expect root_rot in xyzw order!
    root_pos = qpos_seq[:, :3].astype(np.float32)
    root_rot_wxyz = qpos_seq[:, 3:7].astype(np.float32)
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    dof_pos = qpos_seq[:, 7:].astype(np.float32)

    print(
        f"[bold green]Retargeting complete ({args.robot})![/bold green] "
        f"Frames: {qpos_seq.shape[0]}, Duration: {qpos_seq.shape[0] / aligned_fps:.2f}s, "
        f"DoFs: {dof_pos.shape[1]}"
    )

    if args.save_pkl:
        pkl_path = pathlib.Path(args.save_pkl)
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        motion_data = {
            "fps": int(aligned_fps),
            "root_pos": root_pos,
            "root_rot": root_rot_xyzw,
            "dof_pos": dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"[green]Saved GMR .pkl motion to {pkl_path}[/green]")

    if args.save_csv:
        csv_path = pathlib.Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_matrix = np.hstack([root_pos, root_rot_xyzw, dof_pos])
        expected_cols = 29 if args.robot == "booster_k1" else 36
        assert csv_matrix.shape[1] == expected_cols, (
            f"Expected {expected_cols} columns for {args.robot}, got {csv_matrix.shape[1]}"
        )
        np.savetxt(csv_path, csv_matrix, delimiter=",", fmt="%.6f")
        print(
            f"[green]Saved {expected_cols}-column BeyondMimic/MJLab .csv motion to {csv_path} "
            f"(shape={csv_matrix.shape})[/green]"
        )

    if args.save_video:
        render_preview_video(
            xml_path=xml_path,
            qpos_seq=qpos_seq,
            fps=int(aligned_fps),
            video_path=pathlib.Path(args.save_video),
            robot=args.robot,
        )


if __name__ == "__main__":
    main()
