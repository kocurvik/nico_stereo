"""Draw FoundationPose's `track_vis` overlay for poses already saved by an arm.

`run_pose_arm.py` runs with debug=0 and writes only `ob_in_cam/`, while Fig. 4
of the paper is built from `track_vis/` renders. This draws them afterwards,
from the saved poses, with exactly the calls FoundationPose's own
`run_demo.py` uses (lines 35-36 and 69-71): the oriented bounding box of the
mesh, `draw_posed_3d_box`, then `draw_xyz_axis` -- at the same 576x324
resolution and with the same scaled intrinsics as `run_pose_arm.py`, on an
INTER_AREA-downscaled image as the delivered renders are. Redrawing the
delivered network arm this way reproduces its renders. Nothing is re-estimated.

    python -m nico_stereo.RGBD_task.render_track_vis --arm las2_m \
        --cells scene_009/rubiks_cube/0 scene_002/scissors/1
    # validation: redraw the delivered network arm somewhere harmless
    python -m nico_stereo.RGBD_task.render_track_vis --arm left --out_root <scratch> ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
from nico_stereo.config import ROOT, FOUNDATIONPOSE_DIR

FP = FOUNDATIONPOSE_DIR
sys.path.insert(0, str(FP))
sys.path.insert(0, str(FP / "mycpp" / "build"))

from Utils import draw_posed_3d_box, draw_xyz_axis  # noqa: E402
import trimesh  # noqa: E402

from nico_stereo.RGBD_task.run_pose_arm import DATE, SHORTER_SIDE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="directory under pose_estimation/results")
    ap.add_argument("--cells", nargs="+", required=True, help="scene/object/frame")
    ap.add_argument("--out_root", default=None, help="default: results/<arm>")
    args = ap.parse_args()

    from nico_stereo.utils import load_dict
    PE = ROOT / "out" / f"out_{DATE}" / "pose_estimation"
    out_root = Path(args.out_root) if args.out_root else PE / "results" / args.arm
    K_full = np.asarray(load_dict(
        ROOT / "out" / f"out_{DATE}" / "cameras_parameters" / "calib_data.npy"
    )["new_K_l"], dtype=np.float64).reshape(3, 3)

    for spec in args.cells:
        scene, obj, frame = spec.split("/")
        i = int(frame)

        # Same frame indexing and preprocessing as run_pose_arm.py.
        rgb_files = sorted((PE / "undistorted_images_NICO" / scene).glob("*_left.png"),
                           key=lambda p: int(p.stem.split("_")[0]))
        bgr = cv2.imread(str(rgb_files[i]))
        H0, W0 = bgr.shape[:2]
        downscale = SHORTER_SIDE / min(H0, W0)
        H, W = int(H0 * downscale), int(W0 * downscale)
        K = K_full.copy()
        K[:2] *= downscale
        # The pose was estimated on an INTER_NEAREST image (run_pose_arm.py), but
        # the delivered renders are drawn on an INTER_AREA one: that background
        # alone reproduces 99 % of their pixels, NEAREST only 11 %. Match them.
        color = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (W, H),
                           interpolation=cv2.INTER_AREA)

        mesh = trimesh.load(str(PE / "3D_models" / f"{obj}_textured.obj"))
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        pose = np.loadtxt(str(PE / "results" / args.arm / scene / obj / "ob_in_cam" / f"{i:03d}.txt"))
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                            transparency=0, is_input_rgb=True)

        out = out_root / scene / obj / "track_vis"
        out.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(str(out / f"{i:03d}.png"), vis)
        print(f"{args.arm} {scene}/{obj}/{i:03d} -> {out / f'{i:03d}.png'}")


if __name__ == "__main__":
    main()
