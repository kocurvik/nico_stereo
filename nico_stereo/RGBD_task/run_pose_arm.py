"""Run FoundationPose over the downstream scenes from one depth source.

The three delivered arms (`left`, `zed`, `realsense`) were produced on another
machine by a driver that is not in this repository. This reconstructs it for the
robot-head arms, whose staging is identical apart from which depth directory is
read:

    rgb    pose_estimation/undistorted_images_NICO/<scene>/<NNN>_left.png
    depth  pose_estimation/depth_{nn,sgbm,las2_m,las2_h}/<scene>/depth_png/<i>.png
    mask   pose_estimation/masks/<scene>/<object>/000_left.png
    K      new_K_l from calib_data.npy, scaled to the working resolution

Preprocessing mirrors FoundationPose's own YcbineoatReader with
`shorter_side=324`, which is what the delivered 576x324 track_vis renders imply
was used for all three arms.

Frames are indexed by integer, never by sorted filename: the depth files are
named 0.png..10.png, which sorts as 0,1,10,2,... and would silently pair frame 1
with frame 10.

    python -m nico_stereo.RGBD_task.run_pose_arm --arm sgbm
    python -m nico_stereo.RGBD_task.run_pose_arm --arm nn --out results_check/left
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from nico_stereo.config import ROOT, FOUNDATIONPOSE_DIR

FP = FOUNDATIONPOSE_DIR
sys.path.insert(0, str(FP))
sys.path.insert(0, str(FP / "mycpp" / "build"))

from estimater import (  # noqa: E402
    FoundationPose, ScorePredictor, PoseRefinePredictor, dr, trimesh,
    set_logging_format, set_seed, logging)

DATE = "24042026"
SHORTER_SIDE = 324
SCENES = ["scene_001", "scene_002", "scene_004", "scene_005", "scene_006", "scene_009"]
OBJECTS = ["apple", "chips_box", "lemon", "orange", "rubiks_cube", "scissors", "wood_block"]
DEPTH_DIRS = {"nn": "depth_nn", "sgbm": "depth_sgbm",
              "las2_m": "depth_las2_m", "las2_h": "depth_las2_h"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=sorted(DEPTH_DIRS), required=True)
    p.add_argument("--out", default=None,
                   help="results subpath; default results/<arm>")
    p.add_argument("--est_refine_iter", type=int, default=5)
    p.add_argument("--track_refine_iter", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    set_logging_format()
    set_seed(0)

    PE = ROOT / "out" / f"out_{DATE}" / "pose_estimation"
    out_root = PE / (args.out or f"results/{args.arm}")
    depth_root = PE / DEPTH_DIRS[args.arm]

    from nico_stereo.utils import load_dict
    K_full = np.asarray(load_dict(
        ROOT / "out" / f"out_{DATE}" / "cameras_parameters" / "calib_data.npy"
    )["new_K_l"], dtype=np.float64).reshape(3, 3)

    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    for obj in OBJECTS:
        mesh = trimesh.load(str(PE / "3D_models" / f"{obj}_textured.obj"))
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                             mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx,
                             debug=0, debug_dir=str(FP / "debug"))

        for scene in SCENES:
            mask_path = PE / "masks" / scene / obj / "000_left.png"
            if not mask_path.is_file():
                logging.info(f"skip {scene}/{obj}: no mask")
                continue

            rgb_files = sorted((PE / "undistorted_images_NICO" / scene).glob("*_left.png"),
                               key=lambda p: int(p.stem.split("_")[0]))
            n = len(rgb_files)
            depth_files = [depth_root / scene / "depth_png" / f"{i}.png" for i in range(n)]
            missing = [d for d in depth_files if not d.is_file()]
            if missing:
                raise FileNotFoundError(f"{scene}/{obj}: missing depth {missing[0]}")

            H0, W0 = cv2.imread(str(rgb_files[0])).shape[:2]
            downscale = SHORTER_SIDE / min(H0, W0)
            H, W = int(H0 * downscale), int(W0 * downscale)
            K = K_full.copy()
            K[:2] *= downscale

            pose_dir = out_root / scene / obj / "ob_in_cam"
            pose_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n):
                bgr = cv2.imread(str(rgb_files[i]))
                color = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (W, H),
                                   interpolation=cv2.INTER_NEAREST)
                depth = cv2.imread(str(depth_files[i]), -1) / 1e3
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
                depth[depth < 0.001] = 0

                if i == 0:
                    m = cv2.imread(str(mask_path), -1)
                    if m.ndim == 3:
                        m = m[..., 0]
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=m,
                                        iteration=args.est_refine_iter)
                else:
                    pose = est.track_one(rgb=color, depth=depth, K=K,
                                         iteration=args.track_refine_iter)

                np.savetxt(str(pose_dir / f"{i:03d}.txt"), pose.reshape(4, 4))

            logging.info(f"done {args.arm} {scene}/{obj}: {n} frames -> {pose_dir}")


if __name__ == "__main__":
    main()
