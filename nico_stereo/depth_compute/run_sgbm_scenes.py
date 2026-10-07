"""SGBM depth for the six downstream_task scenes, in the `depth_nn` layout.

`run_sgbm.py` covers the 215-frame evaluation set. The pose experiment uses a
different set -- the six `downstream_task` scenes, 11 frames each -- whose
network depth lives in `pose_estimation/depth_nn/`. This writes the SGBM
equivalent to `pose_estimation/depth_sgbm/` so the pose driver can swap one
directory and change nothing else.

Identical matcher, identical a-priori parameters, identical disparity-to-depth
conversion as run_sgbm.py -- it imports them rather than restating them, so the
two cannot drift apart.

Run (from the repository root):  python -m nico_stereo.depth_compute.run_sgbm_scenes
"""
from __future__ import annotations

import json
import logging
import time as time_module
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from nico_stereo.config import ROOT
from nico_stereo.depth_compute.run_sgbm import (
    build_matcher, compute_disparity, describe_matcher, warmup)
from nico_stereo.depth_compute.depth_utils import load_calibration, disparity_to_depth
from nico_stereo.image import load_l_r_images_rectified
from nico_stereo.utils import load_dict

DATE = "24042026"
SCALE = 6
MIN_DISP = 1e-6


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = ROOT
    scenes_dir = root / "datasets" / f"dataset_{DATE}" / "downstream_task"
    calib_file = root / "out" / f"out_{DATE}" / "cameras_parameters" / "calib_data.npy"
    out_root = root / "out" / f"out_{DATE}" / "pose_estimation" / "depth_sgbm"

    logging.info("Matcher: %s", describe_matcher())
    calib_dict = load_dict(calib_file)
    matcher = build_matcher()

    scenes = sorted([d.name for d in scenes_dir.iterdir() if d.is_dir()],
                    key=lambda s: int(s.split("_")[-1]))

    for scene in scenes:
        imgs_l, imgs_r = load_l_r_images_rectified(calib_dict, scenes_dir / scene / "rgb")
        if not imgs_l:
            raise RuntimeError(f"no stereo pairs in {scene}")

        full = imgs_l[0].get_img()
        small = imgs_l[0].get_small_img(scale=SCALE)
        input_resolution = (full.shape[1], full.shape[0])
        inference_resolution = (small.shape[1], small.shape[0])

        K, baseline_m, _ = load_calibration(
            calib_dict_file=calib_file,
            input_resolution=input_resolution,
            inference_resolution=inference_resolution)
        fx = float(K[0, 0])

        depth_dir = out_root / scene / "depth"
        png_dir = out_root / scene / "depth_png"
        depth_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)

        warmup(matcher, small, imgs_r[0].get_small_img(scale=SCALE))

        per_image, times = [], []
        for img_l, img_r in zip(imgs_l, imgs_r):
            n = img_l.get_image_number()
            img0 = img_l.get_small_img(scale=SCALE)
            img1 = img_r.get_small_img(scale=SCALE)

            disp, elapsed = compute_disparity(matcher, img0, img1)
            depth = disparity_to_depth(disp=disp, fx=fx, baseline_m=baseline_m,
                                       min_disp=MIN_DISP)
            times.append(elapsed)

            np.save(depth_dir / f"{n}_depth.npy", depth.astype(np.float32))
            # 16-bit millimetres, as depth_images_scenes.py writes for depth_nn.
            png = np.clip(np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0,
                          0, 65535).astype(np.uint16)
            cv2.imwrite(str(png_dir / f"{n}.png"), png)

            valid = np.isfinite(depth)
            per_image.append({
                "image_id": int(n), "time_s": round(elapsed, 4),
                "depth_min": round(float(np.nanmin(depth)), 4),
                "depth_max": round(float(np.nanmax(depth)), 4),
                "depth_mean": round(float(np.nanmean(depth)), 4),
                "valid_px": int(valid.sum()), "total_px": int(depth.size),
            })

        stats = {
            "run_name": "SGBM_stereo", "date": DATE,
            "model_ckpt": describe_matcher(), "network_type": "stereo",
            "input_resolution": list(input_resolution),
            "inference_resolution": list(inference_resolution),
            "scale": SCALE, "n_images": len(per_image),
            "total_time_s": round(float(np.sum(times)), 4),
            "mean_time_s": round(float(np.mean(times)), 4),
            "min_time_s": round(float(np.min(times)), 4),
            "max_time_s": round(float(np.max(times)), 4),
            "per_image_stats": per_image,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(out_root / scene / "run_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        cov = 100.0 * np.mean([p["valid_px"] / p["total_px"] for p in per_image])
        logging.info("%s: %d frames, mean %.4f s, coverage %.1f%%",
                     scene, len(per_image), np.mean(times), cov)


if __name__ == "__main__":
    main()
