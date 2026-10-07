"""Lite Any Stereo V2 (LAS2) over the rectified stereo pairs of the robot head.

Runs the LAS2-M or LAS2-H release model from
https://github.com/TomTomTommi/LiteAnyStereo over the same rectified pairs, at
the same inference resolution and through the same disparity-to-depth
conversion as every other stereo model, and writes the same layout under
``out/out_<date>/depth_estimation/<run_name>/``.

Inference follows the repository's own ``demo.py``: float32, 0-255 input,
``InputPadder(divis_by=32)``, ``max_disp=192``, ``test_mode=True``, no AMP.
192 px of disparity covers depth down to 0.08 m at 640x360, well below the
closest surface in the dataset.

Colour order: the pipeline hands every network BGR images, because
``nico_stereo.image`` leaves its BGR->RGB conversion commented out, and no delivered
run script converts. LAS2 normalises with ImageNet RGB statistics, so
``--rgb`` feeds it RGB as its own demo does. The default (BGR) is the one that
is comparable with the delivered networks.

LAS2 needs ``fasternet_t0``, which the timm in the ``torch`` env (0.9.16)
lacks. A newer timm is installed with ``--no-deps`` into the repo's private
``_vendor`` directory and put first on ``sys.path`` here only, so the env the
other models share is untouched.

Run (from the repository root):  python -m nico_stereo.depth_compute.run_liteanystereo --model_size m
"""
from __future__ import annotations

import argparse
import logging
import sys
import time as time_module
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from nico_stereo.config import ROOT, THIRD_PARTY
from nico_stereo.depth_compute.depth_utils import load_calibration, disparity_to_depth, save_outputs
from nico_stereo.depth_compute.run_stats import RunStats, save_run_stats
from nico_stereo.image import load_l_r_images_rectified
from nico_stereo.prepare_paths import build_paths, prepare_output_dirs
from nico_stereo.utils import load_dict

LAS_REPO = THIRD_PARTY / "LiteAnyStereo"
sys.path.insert(0, str(LAS_REPO))
sys.path.insert(0, str(LAS_REPO / "_vendor"))

from core.models import build_model, load_model_weights  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402

MAX_DISP = 192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LAS2 on the rectified stereo pairs of the robot head.")
    parser.add_argument("--model_size", type=str, default="m", choices=["s", "m", "l", "h"])
    parser.add_argument("--rgb", action="store_true", help="feed RGB, as LAS2's own demo does")
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--max_imgs", type=int, default=None)
    parser.add_argument("--date", type=str, default="24042026")
    parser.add_argument("--min_disp", type=float, default=1e-6)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = f"LiteAnyStereoV2_{args.model_size.upper()}" + ("_rgb" if args.rgb else "")
    return args


def load_model(model_size: str) -> tuple[torch.nn.Module, Path]:
    ckpt_path = LAS_REPO / "checkpoints" / f"LAS2_{model_size.upper()}.pth"
    model = build_model("las2", fnet_pretrained=False, model_size=model_size, max_disp=MAX_DISP)
    load_model_weights(model, torch.load(ckpt_path, map_location="cuda"), strict=True)
    model.cuda().eval()
    logging.info("Loaded model: %s", ckpt_path)
    return model, ckpt_path


def to_tensor(img: np.ndarray, rgb: bool) -> torch.Tensor:
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.as_tensor(img).cuda().float()[None].permute(0, 3, 1, 2)


def warmup_model(model: torch.nn.Module, inference_resolution: tuple[int, int], n_warmup: int = 5) -> None:
    logging.info("Warming up model (%d passes)...", n_warmup)
    w, h = inference_resolution
    dummy = torch.zeros(1, 3, h, w, device="cuda", dtype=torch.float32)
    dummy, _ = InputPadder(dummy.shape, divis_by=32).pad(dummy, dummy)
    for _ in range(n_warmup):
        _ = model(dummy, dummy, max_disp=MAX_DISP, test_mode=True)
    torch.cuda.synchronize()
    logging.info("Warm-up complete.")


def compute_disparity(model: torch.nn.Module, img0: np.ndarray, img1: np.ndarray, rgb: bool) -> tuple[np.ndarray, float]:
    h, w = img0.shape[:2]
    left_t, right_t = to_tensor(img0, rgb), to_tensor(img1, rgb)
    padder = InputPadder(left_t.shape, divis_by=32)
    left_t, right_t = padder.pad(left_t, right_t)

    torch.cuda.synchronize()
    t0 = time_module.perf_counter()
    disp = model(left_t, right_t, max_disp=MAX_DISP, test_mode=True)
    torch.cuda.synchronize()
    elapsed = time_module.perf_counter() - t0

    disp = padder.unpad(disp.float()).cpu().numpy().reshape(h, w)
    return disp.astype(np.float32), elapsed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    assert args.scale >= 1, "scale must be >= 1"

    torch.autograd.set_grad_enabled(False)

    root_dir = ROOT
    img_dir, calib_dict_file, out_dir = build_paths(root_dir, args.date, args.run_name)
    out_dirs = prepare_output_dirs(out_dir, disp_dir=True)

    logging.info("Image dir: %s", img_dir)
    logging.info("Calibration file: %s", calib_dict_file)
    logging.info("Output dir: %s", out_dir)
    logging.info("Model: LAS2-%s, input %s", args.model_size.upper(), "RGB" if args.rgb else "BGR")

    calib_dict = load_dict(calib_dict_file)
    model, ckpt_path = load_model(args.model_size)

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

    stats = RunStats(
        run_name=args.run_name,
        date=args.date,
        model_ckpt=str(ckpt_path),
        network_type="stereo",
        input_resolution=input_resolution,
        inference_resolution=inference_resolution,
        scale=args.scale,
        n_images=len(imgs_l),
    )

    warmup_model(model, inference_resolution, n_warmup=5)

    for img_l, img_r in tqdm(list(zip(imgs_l, imgs_r)), desc=f"LAS2-{args.model_size.upper()}"):
        img_number = img_l.get_image_number()

        img0 = img_l.get_small_img(scale=args.scale)
        img1 = img_r.get_small_img(scale=args.scale)

        disp, elapsed = compute_disparity(model, img0, img1, args.rgb)

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
