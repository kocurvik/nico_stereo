"""Classical semi-global block matching baseline (OpenCV ``StereoSGBM``).

The learned stereo models are compared against each other but never against
the classical method they replaced. This script fills that gap: it runs
OpenCV's SGBM implementation of Hirschmuller's semi-global matching over the
same rectified pairs, at the same inference resolution, through the same
disparity-to-depth conversion, and writes its output into the same layout as
every network, so that ``compute_cauchy_metrics.py`` picks it up unchanged.

Parameters are fixed a priori and no tuning is performed on the evaluation
frames -- the networks are used off the shelf, so the baseline must be too.
``numDisparities`` follows from the geometry rather than from a search: with
f*B = 14.69 px*m at 640x360, the closest surfaces in the dataset (~0.25 m)
sit at 59 px of disparity, so 64 is the smallest multiple of 16 that covers
the observed range. Everything else is the OpenCV reference configuration --
the smoothness penalties from the documented P1 = 8*C*B^2, P2 = 32*C*B^2 rule
and the consistency filters of ``samples/python/stereo_match.py``.

SGBM is CPU-only; no GPU is used or required.

Run (from the repository root):  python -m nico_stereo.depth_compute.run_sgbm
"""
from __future__ import annotations

import argparse
import logging
import time as time_module
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from nico_stereo.config import ROOT
from nico_stereo.depth_compute.depth_utils import load_calibration, disparity_to_depth, save_outputs
from nico_stereo.depth_compute.run_stats import RunStats, save_run_stats
from nico_stereo.image import load_l_r_images_rectified
from nico_stereo.prepare_paths import build_paths, prepare_output_dirs
from nico_stereo.utils import load_dict

# Fixed a priori -- see the module docstring. Do not tune these on the
# evaluation frames.
BLOCK_SIZE = 5
NUM_DISPARITIES = 64          # 4 x 16; covers disparity up to 64 px, i.e. 0.23 m
MIN_DISPARITY = 0
CHANNELS = 3                  # colour input, as in the P1/P2 rule of the docs
UNIQUENESS_RATIO = 10
DISP12_MAX_DIFF = 1
SPECKLE_WINDOW_SIZE = 100
SPECKLE_RANGE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenCV SGBM on the rectified stereo pairs of the robot head."
    )
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--max_imgs", type=int, default=None)
    parser.add_argument("--date", type=str, default="24042026")
    parser.add_argument("--min_disp", type=float, default=1e-6)
    parser.add_argument("--run_name", type=str, default="SGBM_stereo")
    return parser.parse_args()


def build_matcher() -> cv2.StereoSGBM:
    return cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * CHANNELS * BLOCK_SIZE * BLOCK_SIZE,
        P2=32 * CHANNELS * BLOCK_SIZE * BLOCK_SIZE,
        disp12MaxDiff=DISP12_MAX_DIFF,
        uniquenessRatio=UNIQUENESS_RATIO,
        speckleWindowSize=SPECKLE_WINDOW_SIZE,
        speckleRange=SPECKLE_RANGE,
    )


def describe_matcher() -> str:
    return (
        "cv2.StereoSGBM_create("
        f"minDisparity={MIN_DISPARITY}, numDisparities={NUM_DISPARITIES}, "
        f"blockSize={BLOCK_SIZE}, "
        f"P1={8 * CHANNELS * BLOCK_SIZE * BLOCK_SIZE}, "
        f"P2={32 * CHANNELS * BLOCK_SIZE * BLOCK_SIZE}, "
        f"disp12MaxDiff={DISP12_MAX_DIFF}, uniquenessRatio={UNIQUENESS_RATIO}, "
        f"speckleWindowSize={SPECKLE_WINDOW_SIZE}, speckleRange={SPECKLE_RANGE}, "
        f"mode=STEREO_SGBM_MODE_SGBM) @ OpenCV {cv2.__version__}"
    )


def compute_disparity(matcher: cv2.StereoSGBM, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time_module.perf_counter()
    disp16 = matcher.compute(img0, img1)
    elapsed = time_module.perf_counter() - t0

    # SGBM returns 16-bit fixed point; invalid pixels carry (minDisparity - 1).
    disp = disp16.astype(np.float32) / 16.0
    disp[disp < MIN_DISPARITY] = np.nan
    return disp, elapsed


def warmup(matcher: cv2.StereoSGBM, img0: np.ndarray, img1: np.ndarray, n_warmup: int = 5) -> None:
    logging.info("Warming up (%d passes, shape=%s)...", n_warmup, img0.shape[:2])
    for _ in range(n_warmup):
        matcher.compute(img0, img1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    assert args.scale >= 1, "scale must be >= 1"

    root_dir = ROOT
    img_dir, calib_dict_file, out_dir = build_paths(root_dir, args.date, args.run_name)
    out_dirs = prepare_output_dirs(out_dir, disp_dir=True)

    logging.info("Image dir: %s", img_dir)
    logging.info("Calibration file: %s", calib_dict_file)
    logging.info("Output dir: %s", out_dir)
    logging.info("Matcher: %s", describe_matcher())
    logging.info("OpenCV threads: %d (CPU only, no GPU)", cv2.getNumThreads())

    calib_dict = load_dict(calib_dict_file)

    imgs_l, imgs_r = load_l_r_images_rectified(calib_dict, img_dir, max_imgs=args.max_imgs)
    if len(imgs_l) == 0:
        raise RuntimeError("No stereo pairs found from the provided input arguments")

    logging.info("Found %d pairs", len(imgs_l))

    img_full = imgs_l[0].get_img()
    img_small = imgs_l[0].get_small_img(scale=args.scale)

    input_resolution = (img_full.shape[1], img_full.shape[0])
    inference_resolution = (img_small.shape[1], img_small.shape[0])

    K, baseline_m, _ = load_calibration(
        calib_dict_file=calib_dict_file,
        input_resolution=input_resolution,
        inference_resolution=inference_resolution,
    )
    fx = float(K[0, 0])

    logging.info("Scaled fx: %.6f", fx)
    logging.info("Baseline: %.6f m", baseline_m)
    logging.info("f*B: %.4f px*m  ->  %d px of disparity covers depth >= %.3f m",
                 fx * baseline_m, NUM_DISPARITIES, fx * baseline_m / NUM_DISPARITIES)

    matcher = build_matcher()

    stats = RunStats(
        run_name=args.run_name,
        date=args.date,
        model_ckpt=describe_matcher(),
        network_type="stereo",
        input_resolution=input_resolution,
        inference_resolution=inference_resolution,
        scale=args.scale,
        n_images=len(imgs_l),
    )

    warmup(matcher, img_small, imgs_r[0].get_small_img(scale=args.scale))

    for img_l, img_r in tqdm(list(zip(imgs_l, imgs_r)), desc="sgbm"):
        img_number = img_l.get_image_number()

        img0 = img_l.get_small_img(scale=args.scale)
        img1 = img_r.get_small_img(scale=args.scale)

        disp, elapsed = compute_disparity(matcher, img0, img1)

        depth = disparity_to_depth(
            disp=disp,
            fx=fx,
            baseline_m=baseline_m,
            min_disp=args.min_disp,
        )

        stats.per_image_stats.append({
            "image_id": img_number,
            "time_s": round(elapsed, 4),
            "depth_min": round(float(np.nanmin(depth)), 4),
            "depth_max": round(float(np.nanmax(depth)), 4),
            "depth_mean": round(float(np.nanmean(depth)), 4),
            "valid_px": int(np.isfinite(depth).sum()),
            "total_px": int(depth.size),
        })

        save_outputs(
            img_number=img_number,
            img_left=img0,
            depth=depth,
            out_dirs=out_dirs,
            disp=disp,
        )

    save_run_stats(stats, out_dir)


if __name__ == "__main__":
    main()
