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
from scipy.spatial.transform import Rotation as R


# Physical joint limits of Booster K1 (22-DoF) from k1_mjlab.xml / K1_serial.xml
K1_22DOF_LIMITS = np.array([
    [-1.00,  1.00],  # 0: AAHead_yaw
    [-0.35,  0.87],  # 1: Head_pitch
    [-3.31,  1.22],  # 2: ALeft_Shoulder_Pitch
    [-1.74,  1.57],  # 3: Left_Shoulder_Roll
    [-2.27,  2.27],  # 4: Left_Elbow_Pitch
    [-2.44,  0.00],  # 5: Left_Elbow_Yaw
    [-3.31,  1.22],  # 6: ARight_Shoulder_Pitch
    [-1.57,  1.74],  # 7: Right_Shoulder_Roll
    [-2.27,  2.27],  # 8: Right_Elbow_Pitch
    [ 0.00,  2.44],  # 9: Right_Elbow_Yaw
    [-3.00,  2.21],  # 10: Left_Hip_Pitch
    [-0.40,  1.57],  # 11: Left_Hip_Roll
    [-1.00,  1.00],  # 12: Left_Hip_Yaw
    [ 0.00,  2.23],  # 13: Left_Knee_Pitch
    [-0.87,  0.345], # 14: Left_Ankle_Pitch
    [-0.345, 0.345], # 15: Left_Ankle_Roll
    [-3.00,  2.21],  # 16: Right_Hip_Pitch
    [-1.57,  0.40],  # 17: Right_Hip_Roll
    [-1.00,  1.00],  # 18: Right_Hip_Yaw
    [ 0.00,  2.23],  # 19: Right_Knee_Pitch
    [-0.87,  0.345], # 20: Right_Ankle_Pitch
    [-0.345, 0.345], # 21: Right_Ankle_Roll
], dtype=np.float64)


def detect_bvh_format(bvh_path: pathlib.Path) -> str:
    """Auto-detect whether a BVH file uses LAFAN1/Nokov/CMU or Xsens joint naming."""
    text = bvh_path.read_text(encoding="utf-8", errors="ignore")[:4096]
    if "LeftUpLeg" in text or "Spine1" in text:
        return "lafan1"
    return "xsens"


def select_optimal_bvh_window(
    frames: list[dict],
    fps: int = 30,
    max_duration_sec: float = 5.0,
) -> tuple[list[dict], int, int]:
    """
    Universal Smart Active-Clip Selector:
    When a long BVH file (e.g. 30s - 180s mocap take) is provided without manual --start/--end,
    automatically selects the most active, physically feasible continuous window for Booster K1:
      - Skips initial static T-pose / calibration idle frames
      - Avoids floor-sitting, crawling, or unfeasible deep ground squats (Hips Z < 70% standing height)
      - Maximizes balanced whole-body limb activity without mocap teleportation spikes
    """
    n_total = len(frames)
    win_len = max(60, int(round(max_duration_sec * fps)))
    if n_total <= win_len:
        return frames, 0, n_total

    root_key = "Hips" if "Hips" in frames[0] else next(iter(frames[0].keys()))
    hips_z = np.array([float(fr[root_key][0][2]) for fr in frames], dtype=np.float64)
    standing_hips_z = float(np.percentile(hips_z, 85))

    tracked_keys = [
        k for k in ("Hips", "LeftFootMod", "RightFootMod", "LeftHand", "RightHand", "LeftForeArm", "RightForeArm", "Head")
        if k in frames[0]
    ]
    pos_arr = np.stack(
        [[np.asarray(fr[k][0], dtype=np.float64) for k in tracked_keys] for fr in frames],
        axis=0,
    )  # (T, K, 3)
    vel_mag = np.linalg.norm(np.diff(pos_arr, axis=0), axis=-1).mean(axis=-1) * fps  # (T-1,)
    vel_mag = np.concatenate([[vel_mag[0]], vel_mag])

    best_score = -1e9
    best_start = 0
    step_stride = max(1, fps // 3)  # 0.33s stride

    # Skip first 1.0s if clip is long enough (often T-pose calibration)
    min_start = min(fps, max(0, n_total - win_len - 1))
    for s in range(min_start, n_total - win_len + 1, step_stride):
        e = s + win_len
        w_hips_z = hips_z[s:e]
        w_vel = vel_mag[s:e]

        min_z_ratio = float(np.min(w_hips_z) / max(standing_hips_z, 0.3))
        mean_z_ratio = float(np.mean(w_hips_z) / max(standing_hips_z, 0.3))
        mean_v = float(np.mean(w_vel))
        max_v = float(np.max(w_vel))

        # Penalize deep floor drops (< 72% standing hip height) or extreme mocap glitch spikes (> 6.0 m/s)
        posture_penalty = 0.0
        if min_z_ratio < 0.72:
            posture_penalty -= 25.0 * (0.72 - min_z_ratio)
        if max_v > 5.5:
            posture_penalty -= 5.0 * (max_v - 5.5)

        # Reward rich whole-body activity around 0.35 - 2.2 m/s mean keypoint speed and upright posture
        activity_reward = min(mean_v, 2.0) * 2.5 + mean_z_ratio * 2.0
        score = activity_reward + posture_penalty
        if score > best_score:
            best_score = score
            best_start = s

    best_end = best_start + win_len
    print(
        f"[bold cyan]Universal Auto-Window:[/bold cyan] Selected optimal active segment "
        f"frames [{best_start}:{best_end}] ({win_len} frames = {win_len / fps:.2f}s out of {n_total} total frames)"
    )
    return frames[best_start:best_end], best_start, best_end


def reset_root_to_origin_and_plus_x(qpos_seq: np.ndarray) -> np.ndarray:
    """
    Universal Origin & Heading Alignment:
    Translates horizontal root trajectory so frame 0 starts at (X=0, Y=0)
    and rotates the global trajectory around Z so frame 0 faces +X (yaw = 0).
    """
    out = qpos_seq.copy()
    # Subtract initial XY position
    xy0 = out[0, 0:2].copy()
    out[:, 0:2] -= xy0

    # Compute initial yaw from frame 0 quaternion [qw, qx, qy, qz]
    qw, qx, qy, qz = out[0, 3:7]
    r0 = R.from_quat([qx, qy, qz, qw])
    yaw0 = r0.as_euler("zyx", degrees=False)[0]
    r_inv_yaw = R.from_euler("z", -yaw0, degrees=False)

    # Rotate all XY positions around origin by -yaw0
    xyz = out[:, 0:3].copy()
    out[:, 0:3] = r_inv_yaw.apply(xyz)

    # Rotate all root orientations by -yaw0
    quats_xyzw = out[:, [4, 5, 6, 3]]
    r_all = r_inv_yaw * R.from_quat(quats_xyzw)
    quats_new_xyzw = r_all.as_quat()
    out[:, 3] = quats_new_xyzw[:, 3]  # qw
    out[:, 4:7] = quats_new_xyzw[:, 0:3]  # qx, qy, qz
    return out


def enforce_k1_kinematic_safety(
    xml_path: str,
    qpos_seq: np.ndarray,
    robot: str = "booster_k1",
) -> np.ndarray:
    """
    Universal Booster K1 Kinematic Feasibility & Anti-Collision Filter:
      1. Prevents arm-into-torso self-collision (clamps inward Shoulder Roll).
      2. Clamps all 22 joints within safe physical margins [q_min + 0.015, q_max - 0.015].
      3. Applies a 3-tap zero-phase temporal smoothing filter [0.2, 0.6, 0.2] to eliminate IK jitter.
      4. Ensures safe root height (z >= 0.40m) and aligns stance feet cleanly to ground (z = 0.024m).
    """
    safe_qpos = qpos_seq.copy()
    dof = safe_qpos[:, 7:].copy()

    # 1. Arm-torso self-collision guard:
    # Left_Shoulder_Roll (idx 3) negative = outward left, positive = inward into torso
    # Right_Shoulder_Roll (idx 7) positive = outward right, negative = inward into torso
    dof[:, 3] = np.minimum(dof[:, 3], 0.12)
    dof[:, 7] = np.maximum(dof[:, 7], -0.12)

    # 2. Temporal zero-phase smoothing on joints to prevent finite-difference velocity spikes
    if dof.shape[0] >= 5:
        smoothed = dof.copy()
        smoothed[1:-1] = 0.20 * dof[:-2] + 0.60 * dof[1:-1] + 0.20 * dof[2:]
        dof = smoothed

    # 3. Clamp all 22 joints with a 0.015 rad safety margin inside hard MJCF limits
    margin = 0.015
    q_min = K1_22DOF_LIMITS[:, 0] + margin
    q_max = K1_22DOF_LIMITS[:, 1] - margin
    dof = np.clip(dof, q_min[None, :], q_max[None, :])
    safe_qpos[:, 7:] = dof

    # 4. Ground contact alignment & minimum standing root height guard via MuJoCo FK
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    left_name, right_name = "left_foot_link", "right_foot_link"
    target_min_foot_z = 0.024

    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_name)
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_name)

    foot_z_list = []
    for t in range(safe_qpos.shape[0]):
        data.qpos[:] = safe_qpos[t]
        mujoco.mj_forward(model, data)
        lz = data.xpos[left_foot_id, 2]
        rz = data.xpos[right_foot_id, 2]
        foot_z_list.append(min(lz, rz))

    min_foot_z = float(np.percentile(foot_z_list, 5))
    z_offset = target_min_foot_z - min_foot_z
    safe_qpos[:, 2] += z_offset
    # Ensure root Z never drops below physical standing/crouch floor limit (0.40m)
    safe_qpos[:, 2] = np.maximum(safe_qpos[:, 2], 0.40)

    print(
        f"[cyan]Universal K1 Safety Filter ({robot}):[/cyan] Ground z_offset = {z_offset:+.4f} m, "
        f"root_z range = [{safe_qpos[:, 2].min():.3f}, {safe_qpos[:, 2].max():.3f}] m, "
        f"22-DoF limits & anti-collision enforced."
    )
    return safe_qpos


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
        default=True,
        help="Reset initial horizontal XY position and yaw orientation to origin (enabled by default).",
    )
    parser.add_argument(
        "--no-reset-to-zero",
        action="store_true",
        help="Disable automatic origin and heading reset.",
    )
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=10.0,
        help="Maximum clip duration in seconds for universal auto-window selection when --start/--end are omitted (default: 5.0s).",
    )
    parser.add_argument(
        "--full-clip",
        action="store_true",
        help="Disable automatic active window selection and process the entire untrimmed BVH file.",
    )
    args = parser.parse_args()

    bvh_path = pathlib.Path(args.bvh_file)
    if not bvh_path.is_absolute():
        bvh_path = (WORKSPACE_ROOT / bvh_path).resolve()
    if not bvh_path.exists():
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")

    bvh_format = detect_bvh_format(bvh_path) if args.format == "auto" else args.format
    print(f"[bold blue]Loading BVH file:[/bold blue] {bvh_path} (format={bvh_format})")

    if bvh_format == "lafan1":
        lafan1_frames, actual_human_height = load_lafan1_bvh(str(bvh_path), target_fps=args.target_fps)
        if args.start is not None or args.end is not None:
            lafan1_frames = lafan1_frames[args.start : args.end]
        aligned_fps = args.target_fps
        src_human = "bvh_lafan1"
    elif bvh_format == "xsens":
        xsens_args = argparse.Namespace(
            bvh_file=str(bvh_path),
            scale=args.scale,
            start=args.start,
            end=args.end,
            bvh_format="3DSM",
            reset_to_zero=not args.no_reset_to_zero,
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

    # Universal Auto-Window Selection when --start / --end are not manually specified
    if args.start is None and args.end is None and not args.full_clip:
        lafan1_frames, _, _ = select_optimal_bvh_window(
            lafan1_frames,
            fps=aligned_fps,
            max_duration_sec=args.max_duration_sec,
        )

    print(
        f"[cyan]Loaded {len(lafan1_frames)} active frames at {aligned_fps} FPS "
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
    if not args.no_reset_to_zero:
        qpos_seq = reset_root_to_origin_and_plus_x(qpos_seq)
    qpos_seq = enforce_k1_kinematic_safety(xml_path, qpos_seq, robot=args.robot)

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

    stem_clean = bvh_path.stem.lower().replace(" ", "_")
    save_pkl_str = args.save_pkl or f"data/retargeted/k1_{stem_clean}.pkl"
    save_csv_str = args.save_csv or f"data/retargeted/k1_{stem_clean}.csv"

    if save_pkl_str:
        pkl_path = pathlib.Path(save_pkl_str)
        if not pkl_path.is_absolute():
            pkl_path = (WORKSPACE_ROOT / pkl_path).resolve()
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

    if save_csv_str:
        csv_path = pathlib.Path(save_csv_str)
        if not csv_path.is_absolute():
            csv_path = (WORKSPACE_ROOT / csv_path).resolve()
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
        vid_path = pathlib.Path(args.save_video)
        if not vid_path.is_absolute():
            vid_path = (WORKSPACE_ROOT / vid_path).resolve()
        render_preview_video(
            xml_path=xml_path,
            qpos_seq=qpos_seq,
            fps=int(aligned_fps),
            video_path=vid_path,
            robot=args.robot,
        )


if __name__ == "__main__":
    main()
