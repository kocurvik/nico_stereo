"""Export the rectified evaluation frames at inference resolution.

`retime.py` needs the same input every model saw, but rectification is an
omnidir operation that is not inside any timed region and is slow to repeat
(215 pairs of 4K PNGs per model). It is done once here instead, and the
result is written as lossless PNGs at 640x360, so the timing runs read
byte-identical images to what `Image.get_small_img(scale=6)` produces.

    python export_frames.py [n_frames|all]

Default is 30 evenly spaced frames (for a quick check); "all" writes all 215.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets the script run without `pip install -e .`
from nico_stereo.config import ROOT  # noqa: E402
from nico_stereo.image import Image, get_rectify_functions
from nico_stereo.utils import get_l_r_image_fnames, load_dict
from nico_stereo.depth_compute.depth_utils import load_calibration
from nico_stereo.prepare_paths import build_paths

import cv2
import numpy as np

SCALE = 6
DATE = "24042026"

arg = sys.argv[1] if len(sys.argv) > 1 else "30"
OUT = Path(__file__).resolve().parent / ("_frames_all" if arg == "all" else "_frames")
OUT.mkdir(parents=True, exist_ok=True)

img_dir, calib_file, _ = build_paths(ROOT, DATE, "SGBM_stereo")
calib = load_dict(calib_file)
rect_l, rect_r = get_rectify_functions(calib)

fn_l, fn_r = get_l_r_image_fnames(img_dir)
idx = (np.arange(len(fn_l)) if arg == "all"
       else np.linspace(0, len(fn_l) - 1, int(arg)).round().astype(int))

ids = []
for k, i in enumerate(idx):
    il = Image(fn_l[i], rect_l)
    ir = Image(fn_r[i], rect_r)
    n = il.get_image_number()
    ids.append(n)
    cv2.imwrite(str(OUT / f"{n}_left.png"), il.get_small_img(SCALE))
    cv2.imwrite(str(OUT / f"{n}_right.png"), ir.get_small_img(SCALE))
    if (k + 1) % 25 == 0 or k + 1 == len(idx):
        print(f"  {k + 1}/{len(idx)}", flush=True)

small = cv2.imread(str(OUT / f"{ids[0]}_left.png"))
h, w = small.shape[:2]
K, baseline_m, _ = load_calibration(calib_file, (3840, 2160), (w, h))

json.dump({"ids": ids, "width": w, "height": h, "scale": SCALE,
           "K": K.tolist(), "baseline_m": baseline_m},
          open(OUT / "frames.json", "w"), indent=2)
print(f"wrote {len(ids)} pairs at {w}x{h} to {OUT}")
