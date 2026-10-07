"""
compare_pose.py
---------------
Compare FoundationPose 6-D pose estimates from two cameras that observed
the same scene simultaneously.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from nico_stereo.config import ROOT
from nico_stereo.calibration.ChArUco.charuco_relative_pose_pnp_v3 import load_camera_calibration
from nico_stereo.prepare_paths import prepare_depth_comparison_paths


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def scale_intrinsics(k: np.ndarray, from_wh: tuple[int, int], to_wh: tuple[int, int]) -> np.ndarray:
    sx = to_wh[0] / from_wh[0]
    sy = to_wh[1] / from_wh[1]

    k_out = k.copy().astype(np.float64)
    k_out[0, 0] *= sx
    k_out[1, 1] *= sy
    k_out[0, 2] *= sx
    k_out[1, 2] *= sy

    return k_out


def read_pose(path: Path) -> np.ndarray:
    pose = np.loadtxt(path, dtype=np.float64)

    if pose.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose in {path}, got {pose.shape}")

    return pose


def read_transform_yaml(path: Path, direction: str = "cam2_from_cam1") -> np.ndarray:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)

    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open transform file: {path}")

    try:
        node_21 = fs.getNode("T_cam2_cam1").mat()
        node_12 = fs.getNode("T_cam1_cam2").mat()
    finally:
        fs.release()

    if node_21 is None and node_12 is None:
        raise ValueError(f"Neither T_cam2_cam1 nor T_cam1_cam2 found in {path}")

    t21 = np.asarray(node_21, dtype=np.float64) if node_21 is not None else None
    t12 = np.asarray(node_12, dtype=np.float64) if node_12 is not None else None

    if direction == "cam2_from_cam1":
        return t21 if t21 is not None else np.linalg.inv(t12)

    if direction == "cam1_from_cam2":
        return t12 if t12 is not None else np.linalg.inv(t21)

    raise ValueError(f"Unknown direction '{direction}'")


def project_point(point_3d: np.ndarray, k: np.ndarray) -> tuple[float, float]:
    d_zero = np.zeros((1, 5), dtype=np.float64)

    uv, _ = cv2.projectPoints(
        point_3d.reshape(1, 1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        k,
        d_zero,
    )

    return float(uv[0, 0, 0]), float(uv[0, 0, 1])


def collect_pose_files(results_root: Path, camera: str, scene: str, obj: str) -> dict[str, Path]:
    pose_dir = results_root / camera / scene / obj / "ob_in_cam"

    if not pose_dir.exists():
        raise FileNotFoundError(f"Pose directory not found: {pose_dir}")

    return {p.stem: p for p in sorted(pose_dir.glob("*.txt"))}


def rotation_error_deg(T_pred: np.ndarray, T_ref: np.ndarray) -> float:
    R_pred = T_pred[:3, :3]
    R_ref = T_ref[:3, :3]

    R_delta = R_ref.T @ R_pred

    cos_angle = (np.trace(R_delta) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def translation_error_m(T_pred: np.ndarray, T_ref: np.ndarray) -> float:
    return float(np.linalg.norm(T_pred[:3, 3] - T_ref[:3, 3]))


# ──────────────────────────────────────────────────────────────────────────────
# Symmetry-aware rotation helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_T_from_R(R: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def Rx_deg(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def Ry_deg(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)

    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def Rz_deg(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)

    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_symmetry_candidates() -> dict[str, list[tuple[str, np.ndarray]]]:
    I = np.eye(4, dtype=np.float64)

    rot_180 = [
        ("identity", I),
        ("Rx_180", make_T_from_R(Rx_deg(180))),
        ("Ry_180", make_T_from_R(Ry_deg(180))),
        ("Rz_180", make_T_from_R(Rz_deg(180))),
    ]

    cube_rots = [
        ("identity", I),
        ("Rx_90", make_T_from_R(Rx_deg(90))),
        ("Rx_180", make_T_from_R(Rx_deg(180))),
        ("Rx_270", make_T_from_R(Rx_deg(270))),
        ("Ry_90", make_T_from_R(Ry_deg(90))),
        ("Ry_180", make_T_from_R(Ry_deg(180))),
        ("Ry_270", make_T_from_R(Ry_deg(270))),
        ("Rz_90", make_T_from_R(Rz_deg(90))),
        ("Rz_180", make_T_from_R(Rz_deg(180))),
        ("Rz_270", make_T_from_R(Rz_deg(270))),
    ]

    return {
        "apple": cube_rots,
        "lemon": cube_rots,
        "orange": cube_rots,
        "wood_block": cube_rots,
        "chips_box": cube_rots,
        "rubiks_cube": cube_rots,
        "scissors": cube_rots,
    }


SYMMETRY_CANDIDATES = build_symmetry_candidates()


def symmetry_aware_rotation_error_deg(
    T_pred: np.ndarray,
    T_ref: np.ndarray,
    obj: str,
) -> tuple[float, str, np.ndarray]:
    """
    Find the smallest rotation error after applying possible object symmetries.

    T_pred and T_ref are object-to-camera transforms.
    Symmetry is applied in object coordinates by right multiplication:

        T_pred_sym = T_pred @ T_sym
    """
    candidates = SYMMETRY_CANDIDATES.get(obj, [("identity", np.eye(4, dtype=np.float64))])

    best_error = float("inf")
    best_name = "identity"
    best_T = T_pred.copy()

    for sym_name, T_sym in candidates:
        T_pred_sym = T_pred @ T_sym
        err = rotation_error_deg(T_pred_sym, T_ref)

        if err < best_error:
            best_error = err
            best_name = sym_name
            best_T = T_pred_sym

    return best_error, best_name, best_T


# ──────────────────────────────────────────────────────────────────────────────
# CSV and summary helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"[WARN] Nothing to write: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved → {path}")


def _make_summary_row(scene: str, obj: str, rows: list[dict]) -> dict:
    err3 = np.array([r["err_3d_m"] for r in rows])
    err2 = np.array([r["err_2d_px"] for r in rows])

    dx = np.array([r["diff_x_m"] for r in rows])
    dy = np.array([r["diff_y_m"] for r in rows])
    dz = np.array([r["diff_z_m"] for r in rows])

    err_t = np.array([r["err_translation_m"] for r in rows])
    err_r_raw = np.array([r["err_rotation_raw_deg"] for r in rows])
    err_r_sym = np.array([r["err_rotation_deg"] for r in rows])

    bias = np.stack([dx, dy, dz], axis=1)
    mean_bias = bias.mean(axis=0)
    residual = np.linalg.norm(bias - mean_bias, axis=1)

    return {
        "scene": scene,
        "object": obj,
        "n_frames": len(rows),

        "err3d_mean_m": float(err3.mean()),
        "err3d_median_m": float(np.median(err3)),
        "err3d_std_m": float(err3.std()),
        "err3d_max_m": float(err3.max()),

        "err2d_mean_px": float(err2.mean()),
        "err2d_median_px": float(np.median(err2)),
        "err2d_std_px": float(err2.std()),
        "err2d_max_px": float(err2.max()),

        "err_translation_mean_m": float(err_t.mean()),
        "err_translation_median_m": float(np.median(err_t)),
        "err_translation_std_m": float(err_t.std()),
        "err_translation_max_m": float(err_t.max()),

        "err_rotation_raw_mean_deg": float(err_r_raw.mean()),
        "err_rotation_raw_median_deg": float(np.median(err_r_raw)),
        "err_rotation_raw_std_deg": float(err_r_raw.std()),
        "err_rotation_raw_max_deg": float(err_r_raw.max()),

        "err_rotation_mean_deg": float(err_r_sym.mean()),
        "err_rotation_median_deg": float(np.median(err_r_sym)),
        "err_rotation_std_deg": float(err_r_sym.std()),
        "err_rotation_max_deg": float(err_r_sym.max()),

        "bias_x_m": float(mean_bias[0]),
        "bias_y_m": float(mean_bias[1]),
        "bias_z_m": float(mean_bias[2]),
        "bias_norm_m": float(np.linalg.norm(mean_bias)),

        "residual_mean_m": float(residual.mean()),
        "residual_std_m": float(residual.std()),
        "residual_max_m": float(residual.max()),
    }


def _aggregate_summary(summary_rows: list[dict], group_key: str | None) -> list[dict]:
    numeric = [
        "err3d_mean_m",
        "err3d_median_m",
        "err3d_std_m",
        "err3d_max_m",

        "err2d_mean_px",
        "err2d_median_px",
        "err2d_std_px",
        "err2d_max_px",

        "err_translation_mean_m",
        "err_translation_median_m",
        "err_translation_std_m",
        "err_translation_max_m",

        "err_rotation_raw_mean_deg",
        "err_rotation_raw_median_deg",
        "err_rotation_raw_std_deg",
        "err_rotation_raw_max_deg",

        "err_rotation_mean_deg",
        "err_rotation_median_deg",
        "err_rotation_std_deg",
        "err_rotation_max_deg",

        "bias_x_m",
        "bias_y_m",
        "bias_z_m",
        "bias_norm_m",

        "residual_mean_m",
        "residual_std_m",
        "residual_max_m",
    ]

    if group_key is None:
        row = {
            "group": "GLOBAL",
            "n_scene_obj_pairs": len(summary_rows),
            "total_frames": sum(r["n_frames"] for r in summary_rows),
        }

        for col in numeric:
            row[col] = float(np.mean([r[col] for r in summary_rows]))

        return [row]

    groups: dict[str, list[dict]] = {}

    for r in summary_rows:
        groups.setdefault(r[group_key], []).append(r)

    result = []

    for key, grp in sorted(groups.items()):
        row = {
            group_key: key,
            "n_scene_obj_pairs": len(grp),
            "total_frames": sum(r["n_frames"] for r in grp),
        }

        for col in numeric:
            row[col] = float(np.mean([r[col] for r in grp]))

        result.append(row)

    return result


def _print_scene_summary(scene: str, rows: list[dict]) -> None:
    err3 = np.array([r["err_3d_m"] for r in rows])
    err2 = np.array([r["err_2d_px"] for r in rows])
    err_r_raw = np.array([r["err_rotation_raw_deg"] for r in rows])
    err_r_sym = np.array([r["err_rotation_deg"] for r in rows])

    print(f"\n  ┌─ {scene} ({len(rows)} frames across all objects)")
    print(
        f"  │  3-D error [mm]: mean={err3.mean() * 1000:.1f}  "
        f"std={err3.std() * 1000:.1f}  max={err3.max() * 1000:.1f}"
    )
    print(
        f"  │  2-D error [px]: mean={err2.mean():.1f}  "
        f"std={err2.std():.1f}  max={err2.max():.1f}"
    )
    print(
        f"  │  Rotation raw [deg]: mean={err_r_raw.mean():.2f}  "
        f"std={err_r_raw.std():.2f}"
    )
    print(
        f"  │  Rotation min [deg]: mean={err_r_sym.mean():.2f}  "
        f"std={err_r_sym.std():.2f}"
    )
    print(f"  └{'─' * 50}\n")


def _print_global_summary(summary_rows: list[dict], save_dir: Path) -> None:
    err3 = np.array([r["err3d_mean_m"] for r in summary_rows])
    err2 = np.array([r["err2d_mean_px"] for r in summary_rows])
    err_r_raw = np.array([r["err_rotation_raw_mean_deg"] for r in summary_rows])
    err_r_sym = np.array([r["err_rotation_mean_deg"] for r in summary_rows])
    res = np.array([r["residual_mean_m"] for r in summary_rows])

    print(f"\n{'═' * 70}")
    print(f"  GLOBAL SUMMARY  ({len(summary_rows)} scene/object pairs)")
    print(f"{'─' * 70}")
    print(f"  3-D error [mm]:             mean={err3.mean() * 1000:.2f}  std={err3.std() * 1000:.2f}")
    print(f"  2-D error [px]:             mean={err2.mean():.2f}  std={err2.std():.2f}")
    print(f"  Rotation raw [deg]:         mean={err_r_raw.mean():.2f}  std={err_r_raw.std():.2f}")
    print(f"  Rotation min/sym [deg]:     mean={err_r_sym.mean():.2f}  std={err_r_sym.std():.2f}")
    print(f"  Residual bias-free [mm]:    mean={res.mean() * 1000:.2f}  std={res.std() * 1000:.2f}")
    print(f"{'─' * 70}")

    best_obj = min(summary_rows, key=lambda r: r["err3d_mean_m"])
    worst_obj = max(summary_rows, key=lambda r: r["err3d_mean_m"])

    print(f"  Best  pair: {best_obj['scene']}/{best_obj['object']}  ({best_obj['err3d_mean_m'] * 1000:.1f} mm)")
    print(f"  Worst pair: {worst_obj['scene']}/{worst_obj['object']}  ({worst_obj['err3d_mean_m'] * 1000:.1f} mm)")
    print(f"{'═' * 70}")
    print(f"\n[INFO] All outputs saved to: {save_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ──────────────────────────────────────────────────────────────────────────────

def _show_frame(
    frame_id: str,
    obj: str,
    err_3d: float,
    err_2d: float,
    u_gt: float,
    v_gt: float,
    u_xfer: float,
    v_xfer: float,
) -> None:
    canvas = np.zeros((500, 1000, 3), dtype=np.uint8)
    W = canvas.shape[1]

    def txt(msg, y, color=(200, 200, 200), scale=0.85):
        cv2.putText(canvas, msg, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

    txt(f"frame={frame_id}   object={obj}", 45)
    txt(f"3-D error : {err_3d * 1000:.2f} mm", 100, (0, 220, 255))
    txt(f"2-D error : {err_2d:.2f} px", 155, (120, 255, 120))

    cx, cy = W - 200, 300
    dx = int(round(u_xfer - u_gt))
    dy = int(round(v_xfer - v_gt))

    cv2.circle(canvas, (cx, cy), 9, (255, 0, 255), -1)
    cv2.circle(canvas, (cx + dx, cy + dy), 9, (0, 255, 255), -1)
    cv2.line(canvas, (cx, cy), (cx + dx, cy + dy), (255, 255, 255), 1)

    txt("● magenta = direct RGBD projection", 230, (255, 80, 255), 0.65)
    txt("● cyan    = transferred from left camera", 265, (80, 255, 255), 0.65)
    txt("(dot plot in RGBD image plane)", 300, (150, 150, 150), 0.60)

    cv2.imshow("FoundationPose pose comparison", canvas)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to results-dir/comparison_<scene>_<obj>.csv",
    )

    parser.add_argument(
        "--rgbd-camera",
        default="zed",
        help="Sub-folder name used for the RGBD camera results.",
    )

    parser.add_argument(
        "--point-in-object",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Point expressed in object coordinates [m] to track.",
    )

    parser.add_argument(
        "--left-native-wh",
        nargs=2,
        type=int,
        default=[3840, 2160],
        metavar=("W", "H"),
    )

    parser.add_argument(
        "--left-inference-wh",
        nargs=2,
        type=int,
        default=[576, 324],
        metavar=("W", "H"),
    )

    parser.add_argument(
        "--rgbd-native-wh",
        nargs=2,
        type=int,
        default=[1280, 720],
        metavar=("W", "H"),
    )

    parser.add_argument(
        "--rgbd-inference-wh",
        nargs=2,
        type=int,
        default=[576, 324],
        metavar=("W", "H"),
    )

    parser.add_argument(
        "--translation-mm",
        action="store_true",
        default=True,
        help="Divide relative-pose translation by 1000 (mm → m).",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show interactive per-frame visualisation.",
    )

    args = parser.parse_args()

    date = "24042026"
    parent_dir = ROOT
    (
        gt_data_dir,
        relative_pose,
        calib_rgbd_path,
        calib_stereo_path,
        depth_estimation_dir,
        depth_comparison_dir,
    ) = prepare_depth_comparison_paths(parent_dir, date, args.rgbd_camera)

    out_dir = parent_dir / "out" / f"out_{date}"
    results_dir = out_dir / "pose_estimation" / "results"

    def load_k(calib_path: Path) -> np.ndarray:
        fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)

        if not fs.isOpened():
            raise FileNotFoundError(f"Cannot open calibration: {calib_path}")

        try:
            k = fs.getNode("K_new").mat()

            if k is None:
                k = fs.getNode("K").mat()
        finally:
            fs.release()

        if k is None:
            raise ValueError(f"No K_new or K in {calib_path}")

        return np.asarray(k, dtype=np.float64)

    k_left_native, _ = load_camera_calibration(calib_stereo_path, suffix="left")
    k_rgbd_native = load_k(calib_rgbd_path)

    k_left = scale_intrinsics(
        k_left_native,
        tuple(args.left_native_wh),
        tuple(args.left_inference_wh),
    )

    k_rgbd = scale_intrinsics(
        k_rgbd_native,
        tuple(args.rgbd_native_wh),
        tuple(args.rgbd_inference_wh),
    )

    print(f"\n{'─' * 70}")
    print(f"  LEFT  camera: native {args.left_native_wh} → inference {args.left_inference_wh}")
    print(f"  {k_left}")
    print(f"  RGBD  camera: native {args.rgbd_native_wh} → inference {args.rgbd_inference_wh}")
    print(f"  {k_rgbd}")
    print(f"{'─' * 70}\n")

    # DÔLEŽITÉ:
    # Ak chyby vychádzajú extrémne veľké, otestuj aj direction="cam2_from_cam1".
    t_rgbd_from_left = read_transform_yaml(relative_pose, direction="cam1_from_cam2")
    t_rgbd_from_left[:3, 3] /= 1000.0

    print(f"T_rgbd_from_left (translation in m):\n{t_rgbd_from_left}\n")

    save_dir = out_dir / "pose_estimation" / "results_comparison" / args.rgbd_camera
    save_dir.mkdir(parents=True, exist_ok=True)

    scenes = [
        "scene_001",
        "scene_002",
        "scene_004",
        "scene_005",
        "scene_006",
        "scene_009",
    ]

    objects = [
        "apple",
        "chips_box",
        "lemon",
        "orange",
        "rubiks_cube",
        "scissors",
        "wood_block",
    ]

    p_obj_h = np.array([*args.point_in_object, 1.0], dtype=np.float64)

    summary_rows: list[dict] = []
    all_frame_rows: list[dict] = []

    for scene in scenes:
        scene_dir = save_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)

        scene_rows: list[dict] = []

        for obj in objects:
            try:
                left_files = collect_pose_files(results_dir, "left", scene, obj)
                rgbd_files = collect_pose_files(results_dir, args.rgbd_camera, scene, obj)
            except FileNotFoundError as exc:
                print(f"[SKIP] {exc}")
                continue

            common_ids = sorted(set(left_files) & set(rgbd_files), key=int)

            if not common_ids:
                print(f"[SKIP] No common frames — scene={scene}, object={obj}")
                continue

            print(f"[INFO] {len(common_ids):3d} frames — scene={scene}, object={obj}")

            rows: list[dict] = []

            for frame_id in common_ids:
                t_left = read_pose(left_files[frame_id])
                t_rgbd = read_pose(rgbd_files[frame_id])

                point_left = (t_left @ p_obj_h)[:3]
                point_rgbd_gt = (t_rgbd @ p_obj_h)[:3]
                point_rgbd_xfer = (t_rgbd_from_left @ np.append(point_left, 1.0))[:3]

                diff_3d = point_rgbd_xfer - point_rgbd_gt
                err_3d = float(np.linalg.norm(diff_3d))

                u_gt, v_gt = project_point(point_rgbd_gt, k_rgbd)
                u_xfer, v_xfer = project_point(point_rgbd_xfer, k_rgbd)

                err_2d = float(np.hypot(u_xfer - u_gt, v_xfer - v_gt))

                T_rgbd_xfer = t_rgbd_from_left @ t_left
                T_rgbd_gt = t_rgbd

                err_t_m = translation_error_m(T_rgbd_xfer, T_rgbd_gt)

                err_R_raw_deg = rotation_error_deg(T_rgbd_xfer, T_rgbd_gt)

                err_R_min_deg, best_symmetry, T_rgbd_xfer_best = symmetry_aware_rotation_error_deg(
                    T_pred=T_rgbd_xfer,
                    T_ref=T_rgbd_gt,
                    obj=obj,
                )

                row = {
                    "scene": scene,
                    "object": obj,
                    "frame_id": int(frame_id),

                    "err_3d_m": err_3d,
                    "diff_x_m": float(diff_3d[0]),
                    "diff_y_m": float(diff_3d[1]),
                    "diff_z_m": float(diff_3d[2]),

                    "err_2d_px": err_2d,
                    "rgbd_gt_u": u_gt,
                    "rgbd_gt_v": v_gt,
                    "rgbd_xfer_u": u_xfer,
                    "rgbd_xfer_v": v_xfer,

                    "left_x_m": float(point_left[0]),
                    "left_y_m": float(point_left[1]),
                    "left_z_m": float(point_left[2]),

                    "rgbd_gt_x_m": float(point_rgbd_gt[0]),
                    "rgbd_gt_y_m": float(point_rgbd_gt[1]),
                    "rgbd_gt_z_m": float(point_rgbd_gt[2]),

                    "rgbd_xfer_x_m": float(point_rgbd_xfer[0]),
                    "rgbd_xfer_y_m": float(point_rgbd_xfer[1]),
                    "rgbd_xfer_z_m": float(point_rgbd_xfer[2]),

                    "err_translation_m": err_t_m,

                    # Pôvodná chyba bez korekcie symetrie
                    "err_rotation_raw_deg": err_R_raw_deg,

                    # Najmenšia chyba po vyskúšaní symetrických otočení
                    "err_rotation_deg": err_R_min_deg,
                    "best_rotation_symmetry": best_symmetry,
                }

                rows.append(row)

                if args.show:
                    _show_frame(
                        frame_id,
                        obj,
                        err_3d,
                        err_2d,
                        u_gt,
                        v_gt,
                        u_xfer,
                        v_xfer,
                    )

                    key = cv2.waitKey(0)

                    if key in (27, ord("q")):
                        break

            if args.show:
                cv2.destroyAllWindows()

            if not rows:
                continue

            obj_csv = scene_dir / f"{obj}.csv"
            _write_csv(obj_csv, rows)

            scene_rows.extend(rows)
            all_frame_rows.extend(rows)

            summary_rows.append(_make_summary_row(scene, obj, rows))

        if scene_rows:
            _write_csv(scene_dir / f"{scene}_all_objects.csv", scene_rows)

            scene_summary = [r for r in summary_rows if r["scene"] == scene]

            if scene_summary:
                _write_csv(scene_dir / f"{scene}_summary.csv", scene_summary)

            _print_scene_summary(scene, scene_rows)

    if not summary_rows:
        print("[WARN] No data was processed.")
        return

    _write_csv(save_dir / "all_frames.csv", all_frame_rows)
    _write_csv(save_dir / "summary_per_scene_object.csv", summary_rows)

    per_obj = _aggregate_summary(summary_rows, group_key="object")
    _write_csv(save_dir / "summary_per_object.csv", per_obj)

    per_scene = _aggregate_summary(summary_rows, group_key="scene")
    _write_csv(save_dir / "summary_per_scene.csv", per_scene)

    grand = _aggregate_summary(summary_rows, group_key=None)
    _write_csv(save_dir / "summary_global.csv", grand)

    _print_global_summary(summary_rows, save_dir)


if __name__ == "__main__":
    main()