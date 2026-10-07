"""Check the delivered evaluation data against the numbers printed in the paper.

Every value the paper states about depth accuracy, runtime and the 6D pose
comparison is re-derived here from the exports under ``<repo>/out/out_24042026``
and compared with what ``main.tex`` claims. A clean run means the paper and
the data agree; any line marked MISMATCH or BAD means they do not.

Table II is computed on ONE pixel set per frame: ZED valid & RealSense valid &
every evaluated method valid (the Metrics section says so). It is produced by
``analysis/common_support_las2.py`` into
``common_support_las2_per_image.csv``. Adding or removing any model changes
that set, and with it every row of the table.

LiteAnyStereo V2 time comes from ``paper_scripts/retime_las2.json``, its poses from
``analysis/results/pose_adds_las2_per_{cell,frame}.csv``.

Run:  python paper_scripts/verify_data.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as _st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # lets the script run without `pip install -e .`
from nico_stereo.config import ROOT  # noqa: E402
DATA = os.path.join(ROOT, "out", "out_24042026")
ANALYSIS = os.path.join(os.path.dirname(HERE), "analysis", "results")

failures = []


def check(ok, message):
    if not ok:
        failures.append(message)
    return "" if ok else "  <-- MISMATCH"


def report(title, items, prefix):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    for text, ok in items:
        print(f"  [{'OK ' if ok else 'BAD'}] {text}")
        check(ok, f"{prefix}: {text}")


# ---------------------------------------------------------------- Table II --
CS = pd.read_csv(os.path.join(ANALYSIS, "common_support_las2_per_image.csv"))
cs = CS.groupby("model").mean(numeric_only=True)
curves = CS.pivot(index="image_id", columns="model", values="AbsRel")

# name in the CSV -> (retime key, (AbsRel, RMSE, MAE, MedAE, SILog), Time, (90, 95, 99, 99.9 %))
TABLE_II = {
    "UniDepthV2":         ("UniDepth_mono",          (0.2520, 0.1671, 0.1538, 0.1385, 0.0673), 0.068, (1.97, 4.43, 38.73, 99.99)),
    "MoGe-2":             ("MoGe_mono",              (0.0891, 0.0684, 0.0557, 0.0461, 0.0546), 0.250, (12.65, 25.39, 91.87, 100.00)),
    "DA3":                ("DepthAnything3_mono",    (0.1493, 0.1145, 0.0907, 0.0728, 0.1326), 0.065, (8.45, 16.87, 71.70, 99.99)),
    "BridgeDepth (RVC)":  ("BridgeDepth_rvc",        (0.0722, 0.0714, 0.0546, 0.0370, 0.0421), 0.175, (5.74, 21.69, 96.17, 100.00)),
    "BridgeDepth (Mid.)": ("BridgeDepth_middlebury", (0.0725, 0.0719, 0.0549, 0.0372, 0.0424), 0.176, (5.60, 21.41, 96.04, 100.00)),
    "DEFOM-Stereo":       ("DEFOM_stereo",           (0.0741, 0.0731, 0.0559, 0.0375, 0.0433), 0.487, (5.02, 20.19, 95.97, 100.00)),
    "FoundationStereo":   ("FoundationStereo",       (0.0723, 0.0710, 0.0546, 0.0373, 0.0423), 0.824, (5.30, 20.97, 96.44, 100.00)),
    "S2M2":               ("S2M2_stereo",            (0.0743, 0.0736, 0.0562, 0.0381, 0.0434), 0.238, (5.17, 20.10, 95.82, 100.00)),
    "LAS2-M":             ("LiteAnyStereoV2_M",      (0.0717, 0.0708, 0.0539, 0.0368, 0.0438), 0.025, (6.49, 22.88, 96.09, 100.00)),
    "LAS2-H":             ("LiteAnyStereoV2_H",      (0.0723, 0.0716, 0.0546, 0.0370, 0.0428), 0.048, (5.85, 21.94, 95.96, 100.00)),
    "SGBM":               ("SGBM_stereo",            (0.0900, 0.1054, 0.0660, 0.0448, 0.0852), 0.049, (4.17, 15.18, 93.38, 99.97)),
    "RealSense D435i":    (None,                     (0.0165, 0.0232, 0.0103, 0.0064, 0.0305), None,  (75.30, 92.98, 99.54, 100.00)),
}
ACC = ["AbsRel", "RMSE", "MAE", "MedianAE", "SILog"]
COV = ["within_ci90_pct", "within_ci95_pct", "within_ci99_pct", "within_ci999_pct"]
LEARNED = [n for n in TABLE_II if n not in ("SGBM", "RealSense D435i")]
MODELS = [n for n in TABLE_II if n != "RealSense D435i"]
STEREO = ["BridgeDepth (RVC)", "BridgeDepth (Mid.)", "S2M2", "FoundationStereo",
          "DEFOM-Stereo", "LAS2-M", "LAS2-H"]

# Inference time is re-measured by paper_scripts/retime.py -- one machine, one env,
# every model through the timed region of its own run script. It does not
# depend on which pixels are scored.
TIMES = {**json.load(open(os.path.join(HERE, "retime_results.json")))["models"],
         **json.load(open(os.path.join(HERE, "retime_las2.json")))["models"]}
times = {n: TIMES[v[0]]["mean_time_s"] for n, v in TABLE_II.items() if v[0]}

print("=" * 78)
print("Table II   (common pixel set; every cell)")
print("=" * 78)
n_px = CS.groupby("image_id").n_pixels.first()
print(f"  {CS.image_id.nunique()} frames, {n_px.mean():,.0f} common pixels per frame")
for name, (tkey, acc, t_paper, cov) in TABLE_II.items():
    n = int((CS.model == name).sum())
    bad = [f"{c} {cs.loc[name, c]:.4f} != {v}" for c, v in zip(ACC, acc) if abs(cs.loc[name, c] - v) >= 5e-5]
    bad += [f"{c} {cs.loc[name, c]:.2f} != {v}" for c, v in zip(COV, cov) if abs(cs.loc[name, c] - v) >= 0.005]
    if tkey and abs(times[name] - t_paper) > 0.06 * t_paper:   # timing: 6 % tolerance
        bad.append(f"time {times[name]:.4f} != {t_paper}")
    if n != 215:
        bad.append(f"{n} frames, expected 215")
    print(f"  {name:<20} AbsRel {acc[0]:.4f}  data {cs.loc[name, 'AbsRel']:.4f}"
          + check(not bad, f"Table II {name}: {'; '.join(bad)}"))


def best(col, names=LEARNED, fn=min):
    return fn(names, key=lambda n: cs.loc[n, col])


report("Table II bold cells (best learned model per column)", [
    ("AbsRel: LAS2-M", best("AbsRel") == "LAS2-M"),
    ("RMSE: MoGe-2", best("RMSE") == "MoGe-2"),
    ("MAE: LAS2-M", best("MAE") == "LAS2-M"),
    ("MedAE: LAS2-M", best("MedianAE") == "LAS2-M"),
    ("SILog: BridgeDepth (RVC)", best("SILog") == "BridgeDepth (RVC)"),
    ("Time: LAS2-M", min(LEARNED, key=times.get) == "LAS2-M"),
    ("90 % and 95 %: MoGe-2", best(COV[0], fn=max) == "MoGe-2" and best(COV[1], fn=max) == "MoGe-2"),
    ("99 %: FoundationStereo", best(COV[2], fn=max) == "FoundationStereo"),
    ("RealSense highest at 90 / 95 / 99 %",
     all(cs.loc["RealSense D435i", c] > cs.loc[MODELS, c].max() for c in COV[:3])),
], "Table II")

# ---------------------------------------------- Section V-C: results prose --
lm = curves["LAS2-M"]
others = [n for n in STEREO if n != "LAS2-M"]
band = cs.loc[others, "AbsRel"]
report("Section V-C results prose", [
    ("LAS2-M lowest AbsRel (0.0717) and MedAE (0.0368) of all models",
     best("AbsRel", MODELS) == "LAS2-M" and best("MedianAE", MODELS) == "LAS2-M"),
    ("BridgeDepth (RVC) keeps the lowest SILog (0.0421)", best("SILog") == "BridgeDepth (RVC)"),
    ("MoGe-2 lowest RMSE (0.0684)", best("RMSE") == "MoGe-2"),
    ("remaining stereo models between 0.0722 and 0.0743",
     abs(band.min() - 0.0722) < 5e-5 and abs(band.max() - 0.0743) < 5e-5),
    ("that band is about 0.15 cm at the median scene depth of 0.71 m",
     abs((band.max() - band.min()) * 71 - 0.15) < 0.01),
    ("UniDepthV2 and DA3 roughly 3.5 and 2.1x worse than the best model",
     round(curves["UniDepthV2"].mean() / lm.mean(), 1) == 3.5
     and round(curves["DA3"].mean() / lm.mean(), 1) == 2.1),
    ("per-frame sigma 0.1248 and 0.0659, far above every stereo model",
     abs(curves["UniDepthV2"].std(ddof=1) - 0.1248) < 5e-5
     and abs(curves["DA3"].std(ddof=1) - 0.0659) < 5e-5
     and min(curves["UniDepthV2"].std(ddof=1), curves["DA3"].std(ddof=1))
     > 2 * max(curves[n].std(ddof=1) for n in STEREO)),
], "V-C")


# ------------------------------------------------- Section V-C: runtime --
# Ratios are far steadier between runs than the absolute times, so they keep
# a tight tolerance while the absolute values get the 6 % one.
def near(n, v):
    return abs(times[n] - v) <= 0.06 * v


report("Section V-C runtime, the SGBM and RealSense paragraphs, abstract, conclusion", [
    ("LAS2 is the fastest learned model in both variants",
     max(times["LAS2-M"], times["LAS2-H"]) < min(times[n] for n in LEARNED if not n.startswith("LAS2"))),
    ("LAS2-M 0.025 s, LAS2-H 0.048 s", near("LAS2-M", 0.025) and near("LAS2-H", 0.048)),
    ("DA3 and UniDepthV2 at about 0.07 s", all(0.06 <= times[n] <= 0.08 for n in ("DA3", "UniDepthV2"))),
    ("LAS2-M 6.9x faster than BridgeDepth (RVC)",
     abs(times["BridgeDepth (RVC)"] / times["LAS2-M"] - 6.9) < 0.1),
    ("LAS2-M 19x faster than DEFOM-Stereo", abs(times["DEFOM-Stereo"] / times["LAS2-M"] - 19) < 0.5),
    ("LAS2-M 33x faster than FoundationStereo",
     abs(times["FoundationStereo"] / times["LAS2-M"] - 33) < 0.5),
    ("... while more accurate than all of them",
     all(cs.loc["LAS2-M", "AbsRel"] < cs.loc[n, "AbsRel"] for n in ("BridgeDepth (RVC)", "DEFOM-Stereo", "FoundationStereo"))),
    ("LAS2-H slower and less accurate than LAS2-M",
     times["LAS2-H"] > times["LAS2-M"] and cs.loc["LAS2-H", "AbsRel"] > cs.loc["LAS2-M", "AbsRel"]),
    ("SGBM AbsRel 0.0900, 1.26x that of LAS2-M",
     abs(cs.loc["SGBM", "AbsRel"] - 0.0900) < 5e-5
     and abs(cs.loc["SGBM", "AbsRel"] / cs.loc["LAS2-M", "AbsRel"] - 1.26) < 0.005),
    ("SGBM ahead of two of the three monocular models",
     sum(cs.loc["SGBM", "AbsRel"] < cs.loc[n, "AbsRel"] for n in ("UniDepthV2", "DA3", "MoGe-2")) == 2),
    ("LAS2-M on the GPU twice as fast as SGBM on the CPU", abs(times["SGBM"] / times["LAS2-M"] - 2.0) < 0.1),
    ("RealSense AbsRel 0.0165, MedAE 0.0064 m, within the ZED repeatability of 0.0198 m",
     abs(cs.loc["RealSense D435i", "AbsRel"] - 0.0165) < 5e-5
     and abs(cs.loc["RealSense D435i", "MedianAE"] - 0.0064) < 5e-5
     and cs.loc["RealSense D435i", "MedianAE"] < 0.0198),
    ("abstract: best stereo model 0.072 at 0.025 s",
     round(cs.loc["LAS2-M", "AbsRel"], 3) == 0.072 and near("LAS2-M", 0.025)),
    ("conclusion: LAS2-M 0.0717 at 0.025 s, SGBM 0.0900 at 0.049 s",
     abs(cs.loc["LAS2-M", "AbsRel"] - 0.0717) < 5e-5 and near("SGBM", 0.049)),
], "runtime")

# ------------------------------------------------ Section V-D: coverage --
report("Section V-D sensor-aware prose", [
    ("MoGe-2 highest at the 90 and 95 % levels", all(best(c, fn=max) == "MoGe-2" for c in COV[:2])),
    ("the stereo models trail it there",
     all(cs.loc[STEREO, c].max() < cs.loc["MoGe-2", c] for c in COV[:2])),
    ("stereo models about 96 % inside the 99 % interval, within one point of each other",
     round(cs.loc[STEREO, COV[2]].mean()) == 96 and cs.loc[STEREO, COV[2]].max() - cs.loc[STEREO, COV[2]].min() < 1.0),
    ("RealSense 75.30 % at 90 %, best network 12.65 %, RealSense 99.54 % at 99 %",
     abs(cs.loc["RealSense D435i", COV[0]] - 75.30) < 0.005
     and abs(cs.loc[LEARNED, COV[0]].max() - 12.65) < 0.005
     and abs(cs.loc["RealSense D435i", COV[2]] - 99.54) < 0.005),
], "V-D")

# --------------------------------------------------------- SGBM coverage --
# Not in the text any more, but documented in the README: SGBM leaves holes.
sgbm_stats = json.load(open(os.path.join(DATA, "depth_estimation", "SGBM_stereo", "run_stats.json")))
sgbm_valid = np.array([e["valid_px"] for e in sgbm_stats["per_image_stats"]])
sgbm_total = np.array([e["total_px"] for e in sgbm_stats["per_image_stats"]])
sgbm_frame0 = np.load(os.path.join(DATA, "depth_estimation", "SGBM_stereo", "depth", "0_depth.npy"))
report("SGBM coverage (README)", [
    ("21.3 % of the image has no depth value",
     abs(100 * (1 - sgbm_valid.sum() / sgbm_total.sum()) - 21.3) < 0.05),
    ("the leftmost 64 columns are never filled", bool(np.isnan(sgbm_frame0[:, :64]).all())),
], "SGBM")

# ------------------------------------------------------------- Table I -----
print()
print("=" * 78)
print("Table I   (best_distribution_models.json)")
print("=" * 78)
noise = json.load(open(os.path.join(DATA, "cameras_statistic_model",
                                    "best_distribution_models.json")))
for cam, dist_paper, ks_paper in (("realsense", "t", 0.0164),
                                  ("zed", "cauchy", 0.0205)):
    d = noise[cam]
    flags = (check(d["best_distribution"] == dist_paper,
                   f"{cam}: distribution {d['best_distribution']} != {dist_paper}")
             + check(abs(d["ks_stat"] - ks_paper) < 5e-5,
                     f"{cam}: KS {d['ks_stat']:.4f} != {ks_paper}"))
    print(f"  {cam:<10} paper: {dist_paper:<7} KS={ks_paper:.4f}   "
          f"data: {d['best_distribution']:<7} KS={d['ks_stat']:.4f}{flags}")

# ----------------------------------- Table IV and Section VI: pose, ADD-S ---
# LAS2-M is the learned model of Section VI. Its arm was produced by
# nico_stereo/RGBD_task/run_pose_arm.py --arm las2_m and scored by
# analysis/pose_adds_las2.py; the SGBM and RGB-D rows come from
# analysis/pose_adds_comparison.py. Both use ADD-S with the same model-point
# subsamples and extrinsics. Nothing is re-estimated here.
#
# The measure is ADD-S, not the distance between the poses' translation
# components: these CAD models have origins 3-14 cm from their centroids, so a
# symmetry flip moves that origin without moving the object.
cells = pd.concat([pd.read_csv(os.path.join(ANALYSIS, "pose_adds_per_cell.csv")),
                   pd.read_csv(os.path.join(ANALYSIS, "pose_adds_las2_per_cell.csv"))])
A = cells.pivot_table(index=["scene", "object"], columns="pair", values="adds_m") * 100.0
C = cells.pivot_table(index=["scene", "object"], columns="pair", values="centroid_m") * 100.0
frames = pd.concat([pd.read_csv(os.path.join(ANALYSIS, "pose_adds_per_frame.csv")),
                    pd.read_csv(os.path.join(ANALYSIS, "pose_adds_las2_per_frame.csv"))])
LZ, LR, LS = "LAS2-M -> ZED", "LAS2-M -> RealSense", "LAS2-M -> SGBM"
SZ, SR, RZ = "SGBM -> ZED", "SGBM -> RealSense", "RealSense -> ZED"
TABLE_IV = {      # pairing -> (ADD-S, IQR low, IQR high, centroid), cm
    LZ: (2.10, 1.73, 3.89, 4.29),
    SZ: (2.16, 1.74, 3.09, 4.33),
    LR: (2.43, 1.83, 4.59, 4.87),
    SR: (2.41, 1.97, 3.38, 5.14),
    LS: (0.67, 0.36, 0.95, 1.74),
    RZ: (0.47, 0.37, 0.65, 1.39),
}


def _f4(case, pair, col="adds_m"):
    scene, obj, fid = {"agree": ("scene_009", "rubiks_cube", 0),
                       "disagree": ("scene_002", "scissors", 1)}[case]
    row = frames[(frames.scene == scene) & (frames.object == obj)
                 & (frames.frame_id == fid) & (frames.pair == pair)]
    assert len(row) == 1, f"Fig. 4 {case}/{pair}: {len(row)} rows"
    return float(row.iloc[0][col])


_obj = A.index.get_level_values("object")
per_obj = A[LZ].groupby(_obj).median().sort_values(ascending=False)

items = [("38 object-scene cells in every pairing",
          all(A[p].notna().sum() == 38 and C[p].notna().sum() == 38 for p in TABLE_IV))]
for p, (m, lo, hi, cen) in TABLE_IV.items():
    items.append((f"Table IV {p}: {m:.2f}, {lo:.2f}-{hi:.2f}, {cen:.2f}",
                  abs(A[p].median() - m) < 0.005 and abs(A[p].quantile(.25) - lo) < 0.005
                  and abs(A[p].quantile(.75) - hi) < 0.005 and abs(C[p].median() - cen) < 0.005))
items += [
    ("bold: RealSense -> ZED is the tightest pairing, by ADD-S and centroid",
     A[RZ].median() == min(A[p].median() for p in TABLE_IV)
     and C[RZ].median() == min(C[p].median() for p in TABLE_IV)),
    ("SGBM 2.16 cm against 2.10 cm for LAS2-M, a 0.6 mm difference",
     abs((A[SZ].median() - A[LZ].median()) * 10 - 0.6) < 0.05),
    ("not significant, Wilcoxon p = 0.50", abs(_st.wilcoxon(A[LZ], A[SZ]).pvalue - 0.50) < 0.005),
    ("LAS2-M the better of the two on 17 of 38", int((A[LZ] < A[SZ]).sum()) == 17),
    ("LAS2-M and SGBM agree to 0.67 cm, not distinguishable from the RGB-D pair at p = 0.37",
     abs(_st.wilcoxon(A[LS], A[RZ]).pvalue - 0.37) < 0.005),
    ("most of the 2 cm is common to both head pipelines (ZED gap > 3x their mutual gap)",
     A[LZ].median() / A[LS].median() > 3.0),
    ("per object, largest for the orange (3.05) and the Rubik's cube (2.58)",
     list(per_obj.index[:2]) == ["orange", "rubiks_cube"]
     and abs(per_obj["orange"] - 3.05) < 0.005 and abs(per_obj["rubiks_cube"] - 2.58) < 0.005),
    ("abstract and Sec. VI-C: median ADD-S 2.1 cm", round(A[LZ].median(), 1) == 2.1),
    ("conclusion: 2.10 cm and 2.16 cm", round(A[LZ].median(), 2) == 2.10 and round(A[SZ].median(), 2) == 2.16),
    ("Fig. 4 (a)-(b): ZED and LAS2-M agree, ADD-S at most 1.4 cm", _f4("agree", LZ) * 100 <= 1.405),
    ("Fig. 4 (c)-(d): they do not -- ADD-S above 0.1 of the diameter while the RGB-D pair agrees",
     _f4("disagree", LZ, "adds_frac_diam") > 0.1 and _f4("disagree", RZ) * 100 < 0.5),
]
report("Table IV and Section VI: pose consistency, ADD-S over 38 cells", items, "pose (ADD-S)")

print()
if failures:
    print(f"FAILED — {len(failures)} check(s):")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("All checks passed.")
