"""Build the panels of Fig. 4 from FoundationPose `track_vis` renders.

Fig. 4 contrasts a frame on which the ZED M and the predicted depth agree with
one on which they do not, so that the failure mode Section VI-B describes is
visible and not just tabulated.

    agree       scene_009 / rubiks_cube / frame 0
    disagree    scene_002 / scissors / frame 1

The scissors case is deliberately taken from scene_002 rather than the most
distant scene, so that it isolates the difficulty of the object from the
difficulty of distance.

The ZED M and RealSense renders are the delivered ones,
`pose_estimation/results/<cam>/<scene>/<object>/track_vis/`. The predicted-depth
panel is LAS2-M (`results/las2_m`). `run_pose_arm.py` does not write renders,
so those are drawn from the saved poses by
`nico_stereo/RGBD_task/render_track_vis.py` with FoundationPose's
own drawing calls; redrawing the delivered network arm that way reproduces its
renders. Until the switch to LAS2-M this panel was BridgeDepth (RVC),
`results/left`.

Two files per panel:

    pose_<case>_<cam>.png       452x258 window centred on the green wireframe
    pose_<case>_<cam>_crop.png  the crop main.tex includes

The crop windows are fixed. The original square crops were recovered by
template-matching them against the full renders (exact up to PNG
re-encoding); the crops now written keep their width and vertical centre but
are 20 % less tall. The LAS2-M panel uses the same window as the BridgeDepth
panel it replaces.

    python paper_scripts/make_pose_figure.py
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # lets the script run without `pip install -e .`
from nico_stereo.config import ROOT  # noqa: E402
RESULTS = os.path.join(ROOT, "out", "out_24042026", "pose_estimation", "results")
OUT = os.path.join(HERE, "figures")

CROP_W, CROP_H = 452, 258
CAMERAS = {"zed": "zed", "realsense": "realsense", "net": "las2_m"}
CASES = {
    "agree":    ("scene_009", "rubiks_cube", "000.png"),
    "disagree": ("scene_002", "scissors", "001.png"),
}
# (x, y, side) of the original square crop in the full 576x324 render.
SQUARE = {
    ("agree", "zed"):    (302, 156, 168),
    ("agree", "net"):    (258, 135, 162),
    ("disagree", "zed"): (236, 104, 194),
    ("disagree", "net"): (231, 106, 146),
}
# The crops main.tex includes are 20 % less tall than those squares, at the
# same width. The height is taken equally from top and bottom, except for the
# panels listed in CUT_TOP, which lose it from the top only. main.tex sizes
# the panels by width, so the figure shrinks by the same 20 % in height.
HEIGHT_FRAC = 0.8
CUT_TOP = {("agree", "zed")}


def crop_window(x, y, side, top_only=False):
    h = int(round(side * HEIGHT_FRAC))
    return x, y + (side - h if top_only else (side - h) // 2), side, h


def overlay_centre(img: np.ndarray) -> tuple[int, int]:
    """Centroid of the green wireframe FoundationPose draws on the object."""
    b, g, r = (img[:, :, i].astype(np.int16) for i in range(3))
    mask = (g > 120) & (g - r > 60) & (g - b > 60)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("no green overlay found")
    return int(np.median(xs)), int(np.median(ys))


for case, (scene, obj, frame) in CASES.items():
    for suffix, cam in CAMERAS.items():
        path = os.path.join(RESULTS, cam, scene, obj, "track_vis", frame)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape[:2]

        cx, cy = overlay_centre(img)
        x0 = int(np.clip(cx - CROP_W // 2, 0, w - CROP_W))
        y0 = int(np.clip(cy - CROP_H // 2, 0, h - CROP_H))

        name = f"pose_{case}_{suffix}.png"
        cv2.imwrite(os.path.join(OUT, name), img[y0:y0 + CROP_H, x0:x0 + CROP_W])
        print(f"{case:<9} {cam:<10} {scene}/{obj}/{frame}  "
              f"overlay ({cx},{cy})  crop ({x0},{y0})  -> figures/{name}")

        if (case, suffix) in SQUARE:
            sx, sy, cw, ch = crop_window(*SQUARE[(case, suffix)],
                                         top_only=(case, suffix) in CUT_TOP)
            name = f"pose_{case}_{suffix}_crop.png"
            cv2.imwrite(os.path.join(OUT, name), img[sy:sy + ch, sx:sx + cw])
            print(f"{'':<9} {'':<10} crop ({sx},{sy}) {cw}x{ch}px  -> figures/{name}")
