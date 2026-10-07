"""Score the 6D-pose pairings with ADD-S and centroid distance.

Comparing `T[:3,3]` directly would measure the distance between the MESH
ORIGINS, which is not a property of the object. These meshes have origins
far from their centroids (chips_box 13.7 cm, wood_block 11.7 cm), and the
symmetry correction right-multiplies by a pure rotation, so it fixes the
rotation error but leaves translation untouched. When two arms settle on
different symmetry branches the object has not moved, yet the reported
translation jumps by 20-26 cm.

This recomputes the same 38 object-scene cells with:

    centroid   distance between the two transformed object centroids
    ADD-S      mean nearest-neighbour distance between the two transformed
               model point clouds -- the standard symmetry-tolerant measure

Both are invariant to the mesh origin; ADD-S additionally folds in rotation.
Nothing is re-estimated: this reads the delivered poses only.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # lets the script run without `pip install -e .`

from nico_stereo.config import ROOT  # noqa: E402
from nico_stereo.RGBD_task.compare_6D_pose import (  # noqa: E402
    collect_pose_files, read_pose, read_transform_yaml, translation_error_m,
    symmetry_aware_rotation_error_deg)

PE = ROOT / "out" / "out_24042026" / "pose_estimation"
RESULTS = PE / "results"
REL = ROOT / "out" / "out_24042026" / "cameras_parameters" / "relative_pose"

SCENES = ["scene_001", "scene_002", "scene_004", "scene_005", "scene_006", "scene_009"]
OBJECTS = ["apple", "chips_box", "lemon", "orange", "rubiks_cube", "scissors", "wood_block"]
N_QUERY = 4000          # query subsample; the KD-tree keeps every model point


def load_T(name: str) -> np.ndarray:
    T = read_transform_yaml(REL / name, direction="cam1_from_cam2")
    T[:3, 3] /= 1000.0
    return T


T_zed_left = load_T("relative_pose_zed_to_left_v3.yaml")
T_rs_left = load_T("relative_pose_realsense_to_left_v3.yaml")
T_zed_rs = T_zed_left @ np.linalg.inv(T_rs_left)

# SGBM depth is in the same camera frame as the network depth (rectified
# robot-head left), so it transfers to the RGB-D cameras by the same extrinsic.
PAIRS = {
    "network -> ZED":       ("left", "zed", T_zed_left),
    "network -> RealSense": ("left", "realsense", T_rs_left),
    "SGBM -> ZED":          ("sgbm", "zed", T_zed_left),
    "SGBM -> RealSense":    ("sgbm", "realsense", T_rs_left),
    "network -> SGBM":      ("left", "sgbm", np.eye(4)),
    "RealSense -> ZED":     ("realsense", "zed", T_zed_rs),
}
PAIRS = {k: v for k, v in PAIRS.items()
         if (RESULTS / v[0]).is_dir() and (RESULTS / v[1]).is_dir()}

rng = np.random.default_rng(0)
MODELS = {}
for o in OBJECTS:
    V = np.asarray(trimesh.load(str(PE / "3D_models" / f"{o}_textured.obj")).vertices)
    idx = rng.choice(len(V), size=min(N_QUERY, len(V)), replace=False)
    ext = trimesh.load(str(PE / "3D_models" / f"{o}_textured.obj")).extents
    MODELS[o] = (V, V[idx], V.mean(0), float(np.linalg.norm(ext)))


def tf(T, P):
    return (T[:3, :3] @ P.T).T + T[:3, 3]


rows = []
for label, (src, dst, T_ds) in PAIRS.items():
    for scene in SCENES:
        for obj in OBJECTS:
            try:
                sf = collect_pose_files(RESULTS, src, scene, obj)
                dfl = collect_pose_files(RESULTS, dst, scene, obj)
            except FileNotFoundError:
                continue
            V, Q, c, diam = MODELS[obj]
            for fid in sorted(set(sf) & set(dfl), key=int):
                Tx = T_ds @ read_pose(sf[fid])
                Td = read_pose(dfl[fid])
                adds = cKDTree(tf(Td, V)).query(tf(Tx, Q))[0].mean()
                err_r, _, _ = symmetry_aware_rotation_error_deg(
                    T_pred=Tx, T_ref=Td, obj=obj)
                rows.append({
                    "pair": label, "scene": scene, "object": obj, "frame_id": int(fid),
                    "rot_deg": float(err_r),
                    "origin_m": translation_error_m(Tx, Td),
                    "centroid_m": float(np.linalg.norm(tf(Tx, c[None])[0] - tf(Td, c[None])[0])),
                    "adds_m": float(adds),
                    "diameter_m": diam,
                    "adds_frac_diam": float(adds) / diam,
                })

df = pd.DataFrame(rows)
df.to_csv(HERE / "results" / "pose_adds_per_frame.csv", index=False)
cell = (df.groupby(["pair", "scene", "object"])
          .agg(n_frames=("frame_id", "size"),
               origin_m=("origin_m", "mean"),
               centroid_m=("centroid_m", "mean"),
               adds_m=("adds_m", "mean"),
               adds_frac_diam=("adds_frac_diam", "mean"),
               rot_deg=("rot_deg", "mean"))
          .reset_index())
cell.to_csv(HERE / "results" / "pose_adds_per_cell.csv", index=False)

print("=" * 78)
print("Three pairings, 38 object-scene cells, median over cells")
print("=" * 78)
print(f"{'pairing':<24}{'ADD-S':>9}{'IQR':>14}{'centroid':>11}{'rot':>9}{'origin':>10}")
for label in PAIRS:
    t = cell[cell.pair == label]
    iqr = f"{t.adds_m.quantile(.25)*100:.2f}-{t.adds_m.quantile(.75)*100:.2f}"
    print(f"{label:<24}{t.adds_m.median()*100:>7.2f}cm{iqr:>14}"
          f"{t.centroid_m.median()*100:>9.2f}cm{t.rot_deg.median():>7.1f}d"
          f"{t.origin_m.median()*100:>8.2f}cm")

nz = cell[cell.pair == "network -> ZED"].set_index(["scene", "object"])
rz = cell[cell.pair == "RealSense -> ZED"].set_index(["scene", "object"])
common = nz.index.intersection(rz.index)
a, b = nz.loc[common, "adds_m"], rz.loc[common, "adds_m"]
print(f"\npaired cells: {len(common)}")
print(f"ratio network / RealSense floor : "
      f"origin {nz.loc[common,'origin_m'].median()/rz.loc[common,'origin_m'].median():.2f}x   "
      f"centroid {nz.loc[common,'centroid_m'].median()/rz.loc[common,'centroid_m'].median():.2f}x   "
      f"ADD-S {a.median()/b.median():.2f}x")
print(f"Wilcoxon on ADD-S               : p = {stats.wilcoxon(a, b).pvalue:.4f}")
print(f"network worse on                : {int((a > b).sum())} / {len(common)} cells")

sg = cell[cell.pair == "SGBM -> ZED"]
if len(sg):
    sg_i = sg.set_index(["scene", "object"])
    c2 = sg_i.index.intersection(rz.index)
    print(f"SGBM -> ZED vs the same floor   : "
          f"{sg_i.loc[c2, 'adds_m'].median() / rz.loc[c2, 'adds_m'].median():.2f}x")
    c3 = nz.index.intersection(sg_i.index)
    x, y = nz.loc[c3, "adds_m"], sg_i.loc[c3, "adds_m"]
    print(f"network vs SGBM ({len(c3)} paired cells): "
          f"network {x.median() * 100:.2f} cm, SGBM {y.median() * 100:.2f} cm, "
          f"ratio {y.median() / x.median():.2f}x, "
          f"Wilcoxon p = {stats.wilcoxon(x, y).pvalue:.4f}, "
          f"SGBM worse on {int((y > x).sum())}/{len(c3)} cells")

print(f"\nADD-S recall (fraction of frames agreeing within 0.1 x diameter):")
for label in PAIRS:
    s = df[df.pair == label]
    print(f"  {label:<24}{100*(s.adds_frac_diam < 0.1).mean():5.1f} %")

print("\nper object, ADD-S median over cells (cm):")
piv = cell.pivot_table(index="object", columns="pair", values="adds_m") * 100
print(piv.round(2).to_string())
