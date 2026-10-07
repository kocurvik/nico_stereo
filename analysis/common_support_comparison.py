"""Re-score every model on exactly the pixels the RealSense also fills.

``realsense_as_prediction.py`` scores the RealSense on the pixels where both
RGB-D cameras return a depth. That is a smaller and easier set than the one the
networks are scored on -- the networks are dense, so they are judged everywhere
the ZED~M is valid, including the surfaces the RealSense gives up on. Comparing
those two numbers directly would flatter the RealSense.

This script removes the asymmetry: for every frame it builds one common mask,

    ZED valid  &  RealSense-warped valid  &  every network's warped map valid

and evaluates all ten depth sources on that identical set of pixels. The
resulting table is the apples-to-apples version.

The classical SGBM baseline has the same problem in a stronger form: it leaves
no disparity wherever the left-right check fails, and none at all in the
leftmost ``numDisparities`` columns, so it too enters the intersection rather
than being scored on the pixels it happens to fill.

Run:  python analysis/common_support_comparison.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from nico_stereo.config import ROOT  # noqa: E402
from nico_stereo.calibration.ChArUco.charuco_relative_pose_pnp_v3 import (  # noqa: E402
    load_camera_calibration,
)
from nico_stereo.depth_compare.compare_depth_between_cameras import (  # noqa: E402
    _read_transform,
    warp_depth_to_target,
)
from nico_stereo.utils import scale_intrinsics  # noqa: E402
from realsense_as_prediction import (  # noqa: E402
    resize_depth_validity_weighted,
    standard_metrics,
)

DATA = os.path.join(ROOT, "out", "out_24042026")
DEPTH = os.path.join(ROOT, "datasets", "dataset_24042026", "stereo_4k_depth", "depth")
PARAMS = os.path.join(DATA, "cameras_parameters")
PRED = os.path.join(DATA, "depth_estimation")

RGBD_HW = (720, 1280)
SOURCE_HW = (2160, 3840)
FRAME_HW = (360, 640)

CI_LEVELS = [(0.90, "90"), (0.95, "95"), (0.99, "99"), (0.999, "999")]

NETS = {
    "UniDepth_mono": "UniDepthV2", "MoGe_mono": "MoGe-2", "DepthAnything3_mono": "DA3",
    "BridgeDepth_rvc": "BridgeDepth (RVC)", "BridgeDepth_middlebury": "BridgeDepth (Mid.)",
    "DEFOM_stereo": "DEFOM-Stereo", "FoundationStereo": "FoundationStereo", "S2M2_stereo": "S2M2",
    "SGBM_stereo": "SGBM",
}

# ------------------------------------------------------------- calibration --
K_zed, D_zed = load_camera_calibration(os.path.join(PARAMS, "zed_calibration_1280x720.yaml"))
K_rs, D_rs = load_camera_calibration(os.path.join(PARAMS, "realsense_calibration_1280x720.yaml"))
K_left, D_left = load_camera_calibration(os.path.join(PARAMS, "calib_data.npy"), suffix="left")

rel = os.path.join(PARAMS, "relative_pose")
T_zed_left = _read_transform(os.path.join(rel, "relative_pose_zed_to_left_v3.yaml"), "cam1_from_cam2")
T_rs_left = _read_transform(os.path.join(rel, "relative_pose_realsense_to_left_v3.yaml"), "cam1_from_cam2")
T_zed_left[:3, 3] /= 1000.0
T_rs_left[:3, 3] /= 1000.0
T_zed_rs = T_zed_left @ np.linalg.inv(T_rs_left)

gamma = json.load(open(os.path.join(DATA, "cameras_statistic_model",
                                    "best_distribution_models.json")))["zed"]["params"]["scale"]

frame_wh = (FRAME_HW[1], FRAME_HW[0])
K_zed_s = scale_intrinsics(K_zed.copy(), RGBD_HW[::-1], frame_wh)
K_rs_s = scale_intrinsics(K_rs.copy(), RGBD_HW[::-1], frame_wh)
K_left_s = scale_intrinsics(K_left.copy(), SOURCE_HW[::-1], frame_wh)

ids = sorted(int(f.split("_")[0]) for f in os.listdir(DEPTH) if f.endswith("_zed_depth.npy"))
print(f"{len(ids)} frames x {len(NETS) + 1} depth sources\n")

rows = []
for n, i in enumerate(ids):
    gt = resize_depth_validity_weighted(
        np.load(os.path.join(DEPTH, f"{i}_zed_depth.npy")).astype(np.float32), FRAME_HW)
    rs = resize_depth_validity_weighted(
        np.load(os.path.join(DEPTH, f"{i}_realsense_depth.npy")).astype(np.float32), FRAME_HW)

    warped = {}
    warped["RealSense D435i"], _ = warp_depth_to_target(
        source_depth=rs, k_source=K_rs_s, d_source=D_rs,
        k_target=K_zed_s, d_target=D_zed, t_target_source=T_zed_rs,
        source_depth_scale=1.0, target_hw=FRAME_HW)

    for key, label in NETS.items():
        pred = np.load(os.path.join(PRED, key, "depth", f"{i}_depth.npy")).astype(np.float32)
        warped[label], _ = warp_depth_to_target(
            source_depth=pred, k_source=K_left_s, d_source=D_left,
            k_target=K_zed_s, d_target=D_zed, t_target_source=T_zed_left,
            source_depth_scale=1.0, target_hw=FRAME_HW)

    common = np.isfinite(gt) & (gt > 0)
    for w in warped.values():
        common &= np.isfinite(w) & (w > 0)
    if not common.any():
        continue

    g = gt[common].astype(np.float64)
    for label, w in warped.items():
        p = w[common].astype(np.float64)
        m = standard_metrics(g, p)
        # same coverage measure as Table III, on the common support
        eps = (p - g) / np.clip(g, 1e-9, None)
        for ci, tag in CI_LEVELS:
            bound = gamma * scipy_stats.cauchy.ppf(0.5 + ci / 2.0, loc=0, scale=1)
            m[f"within_ci{tag}_pct"] = float(np.mean(np.abs(eps) <= bound) * 100.0)
        rows.append({"image_id": i, "model": label, "n_pixels": int(common.sum()), **m})

    if (n + 1) % 25 == 0:
        print(f"  {n + 1}/{len(ids)}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "results", "common_support_per_image.csv"), index=False)

agg = df.groupby("model").mean(numeric_only=True).drop(columns=["image_id"])
order = ["RealSense D435i"] + list(NETS.values())
agg = agg.loc[order]

print()
print("=" * 84)
print("All depth sources on the identical pixel set (ZED valid & RealSense valid & all nets)")
print("=" * 84)
print(f"{'source':<22}{'AbsRel':>9}{'RMSE':>9}{'MAE':>9}{'MedAE':>9}{'SILog':>9}{'d1':>8}"
      f"{'CI90':>8}{'CI95':>8}{'CI99':>8}{'CI99.9':>8}")
print("-" * 84)
for label in order:
    r = agg.loc[label]
    print(f"{label:<22}{r['AbsRel']:>9.4f}{r['RMSE']:>9.4f}{r['MAE']:>9.4f}"
          f"{r['MedianAE']:>9.4f}{r['SILog']:>9.4f}{r['delta1']:>8.3f}"
          f"{r['within_ci90_pct']:>8.2f}{r['within_ci95_pct']:>8.2f}"
          f"{r['within_ci99_pct']:>8.2f}{r['within_ci999_pct']:>8.2f}")
print("=" * 84)
print(f"\ncommon-support pixels per frame: {agg['n_pixels'].iloc[0]:,.0f}")

# SGBM is a classical baseline, not one of the learned models
learned = [v for v in NETS.values() if v != "SGBM"]
best_net = agg.loc[learned, "AbsRel"].idxmin()
ratio = agg.loc[best_net, "AbsRel"] / agg.loc["RealSense D435i", "AbsRel"]
print(f"best network on this support: {best_net} "
      f"(AbsRel {agg.loc[best_net, 'AbsRel']:.4f}, {ratio:.1f}x the RealSense)")

# Paired comparison over frames: does the RealSense win frame by frame?
piv = df.pivot(index="image_id", columns="model", values="AbsRel")
wins = int((piv["RealSense D435i"] < piv[best_net]).sum())
print(f"RealSense beats {best_net} on {wins} of {len(piv)} frames")
