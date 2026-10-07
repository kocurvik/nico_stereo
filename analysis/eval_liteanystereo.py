"""Score extra depth sources through the paper's ZED~M protocol, out of band.

``compute_cauchy_metrics.py`` scores whatever is in
``DEPTH_ESTIMATION_NETWORKS`` and rewrites ``all_networks_summary.csv``. That
dict also drives ``common_support_comparison.py``, ``make_figures.py`` and
``verify_data.py``, so adding a model there changes published numbers the next
time any of them runs. This script scores the named sources with the very same
functions -- imported, not reimplemented -- and writes to a separate
``metrics_cauchy_extra/`` directory, leaving the delivered outputs untouched.

A delivered model can be passed as a control: its summary must reproduce the
row in ``all_networks_summary.csv``.

Run:  python analysis/eval_liteanystereo.py \
          LiteAnyStereoV2_M LiteAnyStereoV2_H --control S2M2_stereo
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from nico_stereo.config import ROOT  # noqa: E402
from nico_stereo.depth_compare.compute_cauchy_metrics import (  # noqa: E402
    DATE, RGBD_SUFFIX, SOURCE_HW, TARGET_HW, OUTPUT_SUBDIR,
    compute_cauchy_metrics, resize_depth_validity_weighted,
)
from nico_stereo.depth_noise_model.CI_calculation import load_zed_gamma  # noqa: E402
from nico_stereo.calibration.ChArUco.charuco_relative_pose_pnp_v3 import load_camera_calibration  # noqa: E402
from nico_stereo.depth_compare.compare_depth_between_cameras import _read_transform, warp_depth_to_target  # noqa: E402
from nico_stereo.image import load_rgbd_images  # noqa: E402
from nico_stereo.prepare_paths import prepare_depth_comparison_paths  # noqa: E402
from nico_stereo.utils import load_estimated_depth_map, scale_intrinsics  # noqa: E402

SHOW = ["all_AbsRel", "all_RMSE", "all_MAE", "all_MedianAE", "all_SILog", "all_delta1",
        "within_ci90_pct", "within_ci95_pct", "within_ci99_pct", "within_ci999_pct", "median_abs_z"]


def summarise(per_image_df: pd.DataFrame, key: str, gamma: float, ci_levels: str) -> dict:
    # Same aggregation as compute_cauchy_metrics.main().
    summary = {"nn_name": key, "zed_gamma": gamma, "ci_levels": ci_levels}
    for col in per_image_df.columns:
        if col in {"image_id", "nn_name"}:
            continue
        summary[col] = float(
            per_image_df[col].sum()
            if col in {"n_pixels"} or col.startswith("n_pixels_outside_ci")
            else per_image_df[col].mean()
        )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*", help="directory names under depth_estimation/")
    ap.add_argument("--control", default=None, help="a delivered model; must reproduce its summary row")
    args = ap.parse_args()

    root = Path(CODE).parent
    (gt_data_dir, relative_pose_path, calib_rgbd_path,
     calib_stereo_path, depth_estimation_dir, depth_comparison_dir) = prepare_depth_comparison_paths(root, DATE, RGBD_SUFFIX)

    out_dir = depth_comparison_dir / "metrics_cauchy_extra"
    out_dir.mkdir(parents=True, exist_ok=True)

    k_source, d_source = load_camera_calibration(calib_stereo_path, suffix="left")
    k_target, d_target = load_camera_calibration(calib_rgbd_path)
    t_target_source = _read_transform(relative_pose_path, "cam1_from_cam2")
    t_target_source[:3, 3] /= 1000.0
    gamma = load_zed_gamma(root / "out" / f"out_{DATE}" / "cameras_statistic_model" / "best_distribution_models.json", camera="zed")

    gt_frames = load_rgbd_images(gt_data_dir, suffix=RGBD_SUFFIX, max_imgs=None)

    keys = list(args.sources) + ([args.control] if args.control else [])
    from nico_stereo.depth_compare.compute_cauchy_metrics import CI_LEVELS
    summaries = []
    for key in keys:
        preds = load_estimated_depth_map(depth_estimation_dir, key, max_imgs=None)
        assert len(preds) == len(gt_frames), f"{key}: {len(preds)} predictions for {len(gt_frames)} frames"

        frame_size = preds[0].shape[::-1]
        k_source_scaled = scale_intrinsics(k_source.copy(), SOURCE_HW[::-1], frame_size)
        k_target_scaled = scale_intrinsics(k_target.copy(), TARGET_HW[::-1], frame_size)

        rows = []
        for gt_frame, pred in tqdm(zip(gt_frames, preds), desc=key, total=len(preds)):
            depth_gt = resize_depth_validity_weighted(gt_frame.get_depth(), pred.shape)
            pred_warped, _ = warp_depth_to_target(
                source_depth=pred, k_source=k_source_scaled, d_source=d_source,
                k_target=k_target_scaled, d_target=d_target,
                t_target_source=t_target_source, source_depth_scale=1.0,
                target_hw=depth_gt.shape,
            )
            valid = np.isfinite(pred_warped) & (pred_warped > 0) & np.isfinite(depth_gt) & (depth_gt > 0)
            if not valid.any():
                continue
            rows.append({"image_id": gt_frame.get_image_number(), "nn_name": key,
                         **compute_cauchy_metrics(depth_gt, pred_warped, valid, gamma)})

        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{key}_per_image.csv", index=False)
        summaries.append(summarise(df, key, gamma, str(CI_LEVELS)))
        print(f"[{key}] {len(df)} frames scored")

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "extra_summary.csv", index=False)

    delivered = pd.read_csv(depth_comparison_dir / OUTPUT_SUBDIR / "all_networks_summary.csv")
    if args.control:
        mine = summary.set_index("nn_name").loc[args.control]
        ref = delivered.set_index("nn_name").loc[args.control]
        cols = [c for c in ref.index if c in mine.index and isinstance(ref[c], (int, float, np.floating))]
        diff = max(abs(float(mine[c]) - float(ref[c])) for c in cols)
        print(f"\ncontrol {args.control}: max abs diff vs delivered summary over {len(cols)} columns = {diff:.3g}")
        assert diff < 1e-9, "control does not reproduce the delivered summary"

    table = pd.concat([delivered[delivered.nn_name != args.control], summary]).sort_values("all_AbsRel")
    pd.set_option("display.width", 250)
    print("\n" + table[["nn_name", *SHOW]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
