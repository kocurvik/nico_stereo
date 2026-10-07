"""Score the RealSense D435i as if it were one of the evaluated models.

The paper ranks eight networks against the ZED~M reference. That leaves the
scale of the numbers uninterpretable: is AbsRel 0.089 good? This script runs
the RealSense depth map through the *identical* protocol -- back-project with
its own intrinsics, transform into the ZED~M frame, splat, keep pixels valid
in both maps, and average the per-image metrics -- so that a dedicated RGB-D
camera appears as one more row of Table II.

The geometry comes from the original code (``warp_depth_to_target``), not from
a reimplementation, so the comparison is apples to apples.

Two variants are reported:
  resized  the RealSense depth is first resized to 640x360, the resolution at
           which every network's prediction was stored -- the fair comparison.
  native   the 1280x720 depth is splatted directly into the 640x360 target,
           which gives it denser coverage than the networks had.

Run:  python analysis/realsense_as_prediction.py
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from nico_stereo.config import ROOT  # noqa: E402
from nico_stereo.calibration.ChArUco.charuco_relative_pose_pnp_v3 import (  # noqa: E402
    load_camera_calibration,
)
from nico_stereo.depth_compare.compare_depth_between_cameras import (  # noqa: E402
    _read_transform,
    warp_depth_to_target,
)
from nico_stereo.utils import scale_intrinsics  # noqa: E402

DATA = os.path.join(ROOT, "out", "out_24042026")
DEPTH = os.path.join(ROOT, "datasets", "dataset_24042026", "stereo_4k_depth", "depth")
PARAMS = os.path.join(DATA, "cameras_parameters")
METRICS = os.path.join(DATA, "depth_comparison", "zed", "metrics_cauchy")

RGBD_HW = (720, 1280)          # native resolution of both RGB-D cameras
FRAME_HW = (360, 640)          # resolution the stored predictions live at
CI_LEVELS = [(0.90, "90"), (0.95, "95"), (0.99, "99"), (0.999, "999")]


def resize_depth_validity_weighted(depth, target_hw):
    """Verbatim from compute_cauchy_metrics.py -- averages only valid pixels."""
    target_h, target_w = target_hw
    valid = np.isfinite(depth) & (depth > 0)
    weighted = depth.astype(np.float32).copy()
    weighted[~valid] = 0.0
    depth_interp = cv2.resize(weighted, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    valid_interp = cv2.resize(valid.astype(np.float32), (target_w, target_h),
                              interpolation=cv2.INTER_LINEAR)
    out = np.full((target_h, target_w), np.nan, dtype=np.float32)
    keep = valid_interp > 1e-2
    out[keep] = depth_interp[keep] / valid_interp[keep]
    return out


def standard_metrics(gt, pred):
    """Verbatim from compute_cauchy_metrics.py."""
    abs_err = np.abs(pred - gt)
    rel_err = abs_err / np.clip(gt, 1e-9, None)
    sq_err = (pred - gt) ** 2
    ratio = np.maximum(pred / np.clip(gt, 1e-9, None), gt / np.clip(pred, 1e-9, None))
    log_diff = np.log(np.clip(pred, 1e-9, None)) - np.log(np.clip(gt, 1e-9, None))
    silog_var = np.mean(log_diff ** 2) - np.mean(log_diff) ** 2
    return {
        "AbsRel": float(np.mean(rel_err)),
        "RMSE": float(np.sqrt(np.mean(sq_err))),
        "MAE": float(np.mean(abs_err)),
        "MedianAE": float(np.median(abs_err)),
        "SILog": float(np.sqrt(max(silog_var, 0.0))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


# ------------------------------------------------------------- calibration --
K_zed, D_zed = load_camera_calibration(os.path.join(PARAMS, "zed_calibration_1280x720.yaml"))
K_rs, D_rs = load_camera_calibration(os.path.join(PARAMS, "realsense_calibration_1280x720.yaml"))

# Both files store cam1 = the RGB-D camera, cam2 = the left robot camera, and
# "cam1_from_cam2" is the direction the original protocol reads. Chaining
# through the robot's left camera gives the RealSense -> ZED transform.
T_zed_left = _read_transform(os.path.join(PARAMS, "relative_pose", "relative_pose_zed_to_left_v3.yaml"),
                             "cam1_from_cam2")
T_rs_left = _read_transform(os.path.join(PARAMS, "relative_pose", "relative_pose_realsense_to_left_v3.yaml"),
                            "cam1_from_cam2")
T_zed_left[:3, 3] /= 1000.0
T_rs_left[:3, 3] /= 1000.0
T_zed_rs = T_zed_left @ np.linalg.inv(T_rs_left)

baseline_mm = np.linalg.norm(T_zed_rs[:3, 3]) * 1000.0
print(f"RealSense -> ZED M baseline: {baseline_mm:.1f} mm")

gamma = json.load(open(os.path.join(DATA, "cameras_statistic_model",
                                    "best_distribution_models.json")))["zed"]["params"]["scale"]

frame_wh = (FRAME_HW[1], FRAME_HW[0])
K_zed_s = scale_intrinsics(K_zed.copy(), RGBD_HW[::-1], frame_wh)
K_rs_s = scale_intrinsics(K_rs.copy(), RGBD_HW[::-1], frame_wh)

ids = sorted(int(f.split("_")[0]) for f in os.listdir(DEPTH) if f.endswith("_zed_depth.npy"))
print(f"{len(ids)} frames\n")

rows = {"resized": [], "native": []}
eps_all = {"resized": [], "native": []}

for n, i in enumerate(ids):
    gt_raw = np.load(os.path.join(DEPTH, f"{i}_zed_depth.npy")).astype(np.float32)
    rs_raw = np.load(os.path.join(DEPTH, f"{i}_realsense_depth.npy")).astype(np.float32)
    gt = resize_depth_validity_weighted(gt_raw, FRAME_HW)

    variants = {
        # the networks were stored at 640x360, so give the RealSense the same
        "resized": (resize_depth_validity_weighted(rs_raw, FRAME_HW), K_rs_s),
        # ... and, for reference, the full-resolution source
        "native": (rs_raw, K_rs),
    }

    for name, (src, K_src) in variants.items():
        warped, _ = warp_depth_to_target(
            source_depth=src,
            k_source=K_src, d_source=D_rs,
            k_target=K_zed_s, d_target=D_zed,
            t_target_source=T_zed_rs,
            source_depth_scale=1.0,
            target_hw=FRAME_HW,
        )
        valid = (np.isfinite(warped) & (warped > 0)
                 & np.isfinite(gt) & (gt > 0))
        if not valid.any():
            continue
        g, p = gt[valid].astype(np.float64), warped[valid].astype(np.float64)
        m = standard_metrics(g, p)
        m["n_pixels"] = int(valid.sum())
        m["image_id"] = i
        rows[name].append(m)

        eps = (p - g) / np.clip(g, 1e-9, None)
        for ci, label in CI_LEVELS:
            bound = gamma * scipy_stats.cauchy.ppf(0.5 + ci / 2.0, loc=0, scale=1)
            rows[name][-1][f"within_ci{label}_pct"] = float(np.mean(np.abs(eps) <= bound) * 100.0)
        if name == "resized":
            eps_all[name].append(eps.astype(np.float32))

    if (n + 1) % 50 == 0:
        print(f"  {n + 1}/{len(ids)}")

# ---------------------------------------------------------------- report ---
print()
summary = pd.read_csv(os.path.join(METRICS, "all_networks_summary.csv")).set_index("nn_name")
NAMES = {
    "UniDepth_mono": "UniDepthV2", "MoGe_mono": "MoGe-2", "DepthAnything3_mono": "DA3",
    "BridgeDepth_rvc": "BridgeDepth (RVC)", "BridgeDepth_middlebury": "BridgeDepth (Mid.)",
    "DEFOM_stereo": "DEFOM-Stereo", "FoundationStereo": "FoundationStereo", "S2M2_stereo": "S2M2",
}

out = {}
for name, rs in rows.items():
    df = pd.DataFrame(rs)
    out[name] = {c: df[c].mean() for c in df.columns if c not in ("image_id",)}
    out[name]["n_pixels"] = df["n_pixels"].sum()

print("=" * 92)
print("RealSense D435i scored as a prediction against the ZED M reference")
print("=" * 92)
hdr = f"{'model':<22}{'AbsRel':>9}{'RMSE':>9}{'MAE':>9}{'MedAE':>9}{'SILog':>9}{'d1':>8}{'CI99':>8}"
print(hdr)
print("-" * 92)
for key, label in NAMES.items():
    r = summary.loc[key]
    print(f"{label:<22}{r['all_AbsRel']:>9.4f}{r['all_RMSE']:>9.4f}{r['all_MAE']:>9.4f}"
          f"{r['all_MedianAE']:>9.4f}{r['all_SILog']:>9.4f}{r['all_delta1']:>8.3f}"
          f"{r['within_ci99_pct']:>8.2f}")
print("-" * 92)
for name in ("resized", "native"):
    o = out[name]
    print(f"{'RealSense (' + name + ')':<22}{o['AbsRel']:>9.4f}{o['RMSE']:>9.4f}{o['MAE']:>9.4f}"
          f"{o['MedianAE']:>9.4f}{o['SILog']:>9.4f}{o['delta1']:>8.3f}"
          f"{o['within_ci99_pct']:>8.2f}")
print("=" * 92)

df = pd.DataFrame(rows["resized"])
df.to_csv(os.path.join(HERE, "results", "realsense_as_prediction_per_image.csv"), index=False)
print(f"\nper-image metrics -> analysis/realsense_as_prediction_per_image.csv")

eps = np.concatenate(eps_all["resized"])
print(f"\nmean valid pixels per frame: {out['resized']['n_pixels'] / len(ids):,.0f} "
      f"(networks: {summary['n_pixels'].mean() / len(ids):,.0f})")
print(f"median signed relative error: {np.median(eps) * 100:+.2f} %   "
      f"(a systematic offset between the two cameras would show up here)")
print(f"per-frame AbsRel: mean {df['AbsRel'].mean():.4f}, std {df['AbsRel'].std():.4f}, "
      f"min {df['AbsRel'].min():.4f}, max {df['AbsRel'].max():.4f}")
