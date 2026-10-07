"""Does the reconstructed driver reproduce the delivered `left` arm?

`run_pose_arm.py --arm nn --out results_check/left` re-runs the network depth
through the driver rebuilt in this repository. If it reproduces the delivered
`results/left`, the same driver applied to SGBM depth yields an arm that is
comparable to the three delivered ones -- which is the whole point of the
exercise.

Compared with ADD-S, not raw pose difference: these objects are symmetric and
their mesh origins sit far from their centroids, so a symmetry branch flip
moves T[:3,3] by 20+ cm without moving the object at all.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # lets the script run without `pip install -e .`

from nico_stereo.config import ROOT  # noqa: E402
PE = ROOT / "out" / "out_24042026" / "pose_estimation"

DELIVERED = PE / "results" / "left"
REBUILT = PE / "results_check" / "left"

rng = np.random.default_rng(0)
rows = []
for scene_dir in sorted(REBUILT.iterdir()):
    if not scene_dir.is_dir():
        continue
    for obj_dir in sorted(scene_dir.iterdir()):
        obj, scene = obj_dir.name, scene_dir.name
        mesh = trimesh.load(str(PE / "3D_models" / f"{obj}_textured.obj"))
        V = np.asarray(mesh.vertices)
        q = V[rng.choice(len(V), size=min(4000, len(V)), replace=False)]
        diam = float(np.linalg.norm(mesh.extents))

        for f in sorted((obj_dir / "ob_in_cam").glob("*.txt")):
            ref_f = DELIVERED / scene / obj / "ob_in_cam" / f.name
            if not ref_f.is_file():
                continue
            A = np.loadtxt(f).reshape(4, 4)
            B = np.loadtxt(ref_f).reshape(4, 4)
            ta = (A[:3, :3] @ q.T).T + A[:3, 3]
            tb = (B[:3, :3] @ V.T).T + B[:3, 3]
            rows.append({"scene": scene, "object": obj, "frame": int(f.stem),
                         "adds_m": float(cKDTree(tb).query(ta)[0].mean()),
                         "adds_frac_diam": float(cKDTree(tb).query(ta)[0].mean()) / diam})

df = pd.DataFrame(rows)
if df.empty:
    print("no overlapping poses found")
    raise SystemExit(1)

print(f"frames compared: {len(df)}   cells: {df.groupby(['scene','object']).ngroups}")
print(f"ADD-S vs delivered   median {df.adds_m.median()*100:.3f} cm"
      f"   mean {df.adds_m.mean()*100:.3f} cm   max {df.adds_m.max()*100:.3f} cm")
print(f"frames within 0.1 x diameter of the delivered pose: "
      f"{100*(df.adds_frac_diam < 0.1).mean():.1f} %")
worst = df.nlargest(5, "adds_m")
print("\nlargest disagreements:")
for _, r in worst.iterrows():
    print(f"  {r.scene} {r['object']:<12} frame {int(r.frame):3d}   {r.adds_m*100:7.2f} cm")
df.to_csv(HERE / "results" / "validate_driver_per_frame.csv", index=False)
