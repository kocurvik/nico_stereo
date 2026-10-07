"""LAS2 depth for the six downstream_task scenes, in the `depth_nn` layout.

`run_liteanystereo.py` covers the 215-frame evaluation set. The pose experiment
uses the six `downstream_task` scenes, whose network depth lives in
`pose_estimation/depth_nn/`. This writes the LAS2 equivalent to
`pose_estimation/depth_las2_<size>/` so the pose driver can swap one directory
and change nothing else -- exactly as `run_sgbm_scenes.py` does for SGBM.

Model loading and inference are imported from run_liteanystereo.py, so the two
cannot drift apart. BGR input, as for every delivered model.

Run (from the repository root):  python -m nico_stereo.depth_compute.run_liteanystereo_scenes --model_size m
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from nico_stereo.config import ROOT
from nico_stereo.depth_compute.run_liteanystereo import compute_disparity, load_model, warmup_model
from nico_stereo.depth_compute.depth_utils import load_calibration, disparity_to_depth
from nico_stereo.image import load_l_r_images_rectified
from nico_stereo.utils import load_dict

DATE = "24042026"
SCALE = 6
MIN_DISP = 1e-6


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", default="m", choices=["s", "m", "l", "h"])
    args = ap.parse_args()

    torch.autograd.set_grad_enabled(False)

    root = ROOT
    scenes_dir = root / "datasets" / f"dataset_{DATE}" / "downstream_task"
    calib_file = root / "out" / f"out_{DATE}" / "cameras_parameters" / "calib_data.npy"
    out_root = root / "out" / f"out_{DATE}" / "pose_estimation" / f"depth_las2_{args.model_size}"

    calib_dict = load_dict(calib_file)
    model, ckpt_path = load_model(args.model_size)

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

        warmup_model(model, inference_resolution, n_warmup=5)

        per_image, times = [], []
        for img_l, img_r in zip(imgs_l, imgs_r):
            n = img_l.get_image_number()
            img0 = img_l.get_small_img(scale=SCALE)
            img1 = img_r.get_small_img(scale=SCALE)

            disp, elapsed = compute_disparity(model, img0, img1, rgb=False)
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
            "run_name": f"LiteAnyStereoV2_{args.model_size.upper()}", "date": DATE,
            "model_ckpt": str(ckpt_path), "network_type": "stereo",
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
