"""Offline CSV-to-NPZ motion converter for MJLab BeyondMimic tracking on Booster K1 (22-DoF).

Converts a 29-column retargeted Booster K1 CSV file:
  [x, y, z, qx, qy, qz, qw, j_0, ..., j_21]
into MJLab's native .npz motion reference format at 50 FPS containing:
  - fps: [50]
  - joint_pos: (T, 22)
  - joint_vel: (T, 22)
  - body_pos_w: (T, 23, 3)
  - body_quat_w: (T, 23, 4)  [wxyz convention]
  - body_lin_vel_w: (T, 23, 3)
  - body_ang_vel_w: (T, 23, 3)

Also optionally renders the reference trajectory inside MJLab's Booster K1
flat-terrain scene using OffscreenRenderer.
"""

from __future__ import annotations

import os
import pathlib
import sys

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
import imageio
import numpy as np
import torch
from rich import print
from tqdm import tqdm

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.k1.env_cfgs import booster_k1_flat_tracking_env_cfg
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


def convert_csv_to_npz(
    input_csv: pathlib.Path,
    output_npz: pathlib.Path,
    robot_name: str = "booster_k1",
    input_fps: float = 30.0,
    output_fps: float = 50.0,
    device: str = "cuda:0",
    output_video: pathlib.Path | None = None,
) -> dict[str, np.ndarray]:
    if not torch.cuda.is_available() and device.startswith("cuda"):
        device = "cpu"

    print(
        f"[bold blue]Converting CSV to MJLab NPZ ({robot_name}):[/bold blue] {input_csv} -> {output_npz} "
        f"({input_fps} FPS -> {output_fps} FPS on {device})"
    )

    motion = MotionLoader(
        motion_file=str(input_csv),
        input_fps=input_fps,
        output_fps=output_fps,
        device=torch.device(device),
        line_range=None,
    )

    env_cfg = booster_k1_flat_tracking_env_cfg()
    cam_dist = 1.75
    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps

    scene = Scene(env_cfg.scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    renderer = None
    if output_video is not None:
        viewer_cfg = ViewerConfig(
            height=720,
            width=960,
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            entity_name="robot",
            distance=cam_dist,
            elevation=-12.0,
            azimuth=135.0,
        )
        renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
        renderer.initialize()

    robot: Entity = scene["robot"]
    robot_joint_indexes = robot.find_joints(
        env_cfg.actions["joint_pos"].actuator_names, preserve_order=True
    )[0]

    log: dict[str, list[np.ndarray] | list[ int ]] = {
        "fps": [int(round(output_fps))],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    frames = []

    scene.reset()
    file_saved = False
    pbar = tqdm(total=motion.output_frames, desc="Forward Kinematics & Velocity Rollout")

    while not file_saved:
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        if renderer is not None:
            renderer.update(sim.data)
            frames.append(renderer.render())

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy())

        pbar.update(1)
        if reset_flag:
            file_saved = True

    pbar.close()

    np_log: dict[str, np.ndarray] = {"fps": np.asarray(log["fps"], dtype=np.int64)}
    for k in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        np_log[k] = np.stack(log[k], axis=0)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **np_log)
    print(f"[bold green]Saved MJLab motion reference NPZ to {output_npz}[/bold green]")
    for k, v in np_log.items():
        print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")

    if renderer is not None and output_video is not None:
        renderer.close()
        output_video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_video), frames, fps=int(round(output_fps)), macro_block_size=1)
        print(f"[green]Saved MJLab reference motion video to {output_video}[/green]")

    return np_log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert 36-column retargeted CSV to MJLab .npz motion format."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Input 36-column CSV file produced by GMR retargeting.",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        required=True,
        help="Output .npz path for MJLab MotionCommand.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="booster_k1",
        choices=["booster_k1"],
        help="Target robot model (booster_k1).",
    )
    parser.add_argument(
        "--input-fps",
        type=float,
        default=30.0,
        help="Frame rate of the input CSV (default: 30.0).",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=50.0,
        help="Target control frame rate for MJLab (default: 50.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Simulation device (default: cuda:0).",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Optional output .mp4 path for rendered MJLab reference trajectory.",
    )
    args = parser.parse_args()

    convert_csv_to_npz(
        input_csv=pathlib.Path(args.input_csv).resolve(),
        output_npz=pathlib.Path(args.output_npz).resolve(),
        robot_name=args.robot,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        device=args.device,
        output_video=pathlib.Path(args.output_video).resolve() if args.output_video else None,
    )


if __name__ == "__main__":
    main()
