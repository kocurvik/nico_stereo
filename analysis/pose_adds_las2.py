"""ADD-S for the LiteAnyStereo V2 pose arms, beside the delivered pairings.

Same measure, same extrinsics and the same model-point subsamples (seed 0,
same object order) as pose_adds_comparison.py. That script is deliberately not
modified -- it writes the CSVs behind Table IV and verify_data.py -- and the
delivered pairings are read from its outputs rather than recomputed.

The LAS2 arms ran through the local reconstruction of the pose driver, while
the delivered network arm (`results/left`) was produced on the original author's machine; the two
drivers agree only to a median 0.17 cm ADD-S on identical depth. So the LAS2
arms are compared against `results_check/left` as well -- the same BridgeDepth
depth re-run through the local driver -- which isolates the depth from the
driver. Nothing is re-estimated here.

Run:  python analysis/pose_adds_las2.py
      (after run_pose_arm.py --arm las2_m / --arm las2_h)
"""
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
    collect_pose_files, read_pose, read_transform_yaml)

PE = ROOT / "out" / "out_24042026" / "pose_estimation"
RESULTS = PE / "results"
REL = ROOT / "out" / "out_24042026" / "cameras_parameters" / "relative_pose"

SCENES = ["scene_001", "scene_002", "scene_004", "scene_005", "scene_006", "scene_009"]
OBJECTS = ["apple", "chips_box", "lemon", "orange", "rubiks_cube", "scissors", "wood_block"]
N_QUERY = 4000
LOCAL_NET = "../results_check/left"     # collect_pose_files joins RESULTS / arm


def load_T(name: str) -> np.ndarray:
    T = read_transform_yaml(REL / name, direction="cam1_from_cam2")
    T[:3, 3] /= 1000.0
    return T


T_zed_left = load_T("relative_pose_zed_to_left_v3.yaml")
T_rs_left = load_T("relative_pose_realsense_to_left_v3.yaml")

ARMS = {k: v for k, v in {"las2_m": "LAS2-M", "las2_h": "LAS2-H"}.items()
        if (RESULTS / k).is_dir()}
PAIRS = {
    "network (local) -> ZED":       (LOCAL_NET, "zed", T_zed_left),
    "network (local) -> RealSense": (LOCAL_NET, "realsense", T_rs_left),
}
for arm, name in ARMS.items():
    PAIRS[f"{name} -> ZED"] = (arm, "zed", T_zed_left)
    PAIRS[f"{name} -> RealSense"] = (arm, "realsense", T_rs_left)
    PAIRS[f"network -> {name}"] = ("left", arm, np.eye(4))
    PAIRS[f"{name} -> SGBM"] = (arm, "sgbm", np.eye(4))   # Table IV's direction

# Identical subsamples to pose_adds_comparison.py: same seed, same order.
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
                rows.append({
                    "pair": label, "scene": scene, "object": obj, "frame_id": int(fid),
                    "centroid_m": float(np.linalg.norm(tf(Tx, c[None])[0] - tf(Td, c[None])[0])),
                    "adds_m": float(adds), "diameter_m": diam,
                    "adds_frac_diam": float(adds) / diam,
                })

df = pd.DataFrame(rows)
df.to_csv(HERE / "results" / "pose_adds_las2_per_frame.csv", index=False)
cell = (df.groupby(["pair", "scene", "object"])
          .agg(n_frames=("frame_id", "size"), centroid_m=("centroid_m", "mean"),
               adds_m=("adds_m", "mean"), adds_frac_diam=("adds_frac_diam", "mean"))
          .reset_index())
cell.to_csv(HERE / "results" / "pose_adds_las2_per_cell.csv", index=False)

# Delivered pairings, as pose_adds_comparison.py wrote them.
ref_cell = pd.read_csv(HERE / "results" / "pose_adds_per_cell.csv")
ref_frame = pd.read_csv(HERE / "results" / "pose_adds_per_frame.csv")
all_cell = pd.concat([ref_cell[cell.columns.intersection(ref_cell.columns)], cell])
all_frame = pd.concat([ref_frame[["pair", "adds_frac_diam"]], df[["pair", "adds_frac_diam"]]])

print("=" * 88)
print("Median over object-scene cells")
print("=" * 88)
print(f"{'pairing':<30}{'cells':>6}{'ADD-S':>10}{'IQR':>14}{'centroid':>11}{'recall@0.1d':>13}")
for label in [*ref_cell.pair.unique(), *PAIRS]:
    t = all_cell[all_cell.pair == label]
    iqr = f"{t.adds_m.quantile(.25) * 100:.2f}-{t.adds_m.quantile(.75) * 100:.2f}"
    rec = 100 * (all_frame[all_frame.pair == label].adds_frac_diam < 0.1).mean()
    print(f"{label:<30}{len(t):>6}{t.adds_m.median() * 100:>8.2f}cm{iqr:>14}"
          f"{t.centroid_m.median() * 100:>9.2f}cm{rec:>11.1f} %")


def paired(a_label, b_label):
    a = all_cell[all_cell.pair == a_label].set_index(["scene", "object"]).adds_m
    b = all_cell[all_cell.pair == b_label].set_index(["scene", "object"]).adds_m
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    print(f"  {a_label:<22} vs {b_label:<24} ({len(common)} cells): "
          f"{a.median() * 100:.2f} vs {b.median() * 100:.2f} cm, ratio {a.median() / b.median():.2f}x, "
          f"Wilcoxon p = {stats.wilcoxon(a, b).pvalue:.4f}, first better on {int((a < b).sum())}/{len(common)}")


print("\nDriver effect (same BridgeDepth depth, local vs delivered driver):")
paired("network (local) -> ZED", "network -> ZED")
print("\nPaired over cells (ADD-S):")
for name in ARMS.values():
    paired(f"{name} -> ZED", "network (local) -> ZED")
    paired(f"{name} -> ZED", "network -> ZED")
    paired(f"{name} -> ZED", "SGBM -> ZED")
    paired(f"{name} -> ZED", "RealSense -> ZED")
    paired(f"{name} -> RealSense", "network (local) -> RealSense")

print("\nPer object, median ADD-S over cells against the ZED M (cm):")
zed_pairs = ["network -> ZED", "network (local) -> ZED", "SGBM -> ZED",
             *[f"{n} -> ZED" for n in ARMS.values()]]
piv = all_cell[all_cell.pair.isin(zed_pairs)].pivot_table(
    index="object", columns="pair", values="adds_m", aggfunc="median") * 100
print(piv[zed_pairs].round(2).to_string())
