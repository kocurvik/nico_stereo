"""Regenerate the result figures of the paper with English labels.

All numbers are read from the evaluation outputs of the underlying diploma
thesis (M. Palider, 2026), which are expected under ``<repo>/out/out_24042026``
(a directory junction to the delivered ``nico_stereo`` data is fine).

Run:  python paper_scripts/make_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # lets the script run without `pip install -e .`
from nico_stereo.config import ROOT  # noqa: E402
OUT = os.path.join(HERE, "figures")
DATA = os.path.join(ROOT, "out", "out_24042026")
METRICS = os.path.join(DATA, "depth_comparison", "zed", "metrics_cauchy")
os.makedirs(OUT, exist_ok=True)

# Typeset every label through LaTeX with the same font the document uses.
# Matplotlib's own "Times New Roman" is the Monotype TrueType face, whereas
# IEEEtran sets the body in URW NimbusRomNo9L; the two are metric-compatible
# but visibly different, which showed up in the figure labels. Loading
# `times` here reproduces IEEEtran's exact combination of Nimbus Roman text
# and Computer Modern math.
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathptmx}",
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
})

# Categorical palette, validated for CVD separation (deutan/tritan dE >= 15).
C_STEREO = "#1f4e9c"
C_MONO = "#c0392b"
C_CLASSIC = "#4d4d4d"

# short key -> the name used in the paper's tables and text
PAPER_NAMES = {"UniDepth": "UniDepthV2", "MoGe": "MoGe-2", "DA3": "DA3"}

# paper label -> key used in the evaluation outputs
MONO = {"UniDepth": "UniDepth_mono",
        "MoGe": "MoGe_mono",
        "DA3": "DepthAnything3_mono"}
STEREO = {"BridgeDepth (RVC)": "BridgeDepth_rvc",
          "BridgeDepth (Mid.)": "BridgeDepth_middlebury",
          "S2M2": "S2M2_stereo",
          "FoundationStereo": "FoundationStereo",
          "DEFOM-Stereo": "DEFOM_stereo"}
CLASSIC = {"SGBM": "SGBM_stereo"}              # not learned; its own category
ALL_MODELS = {**MONO, **STEREO, **CLASSIC}
PAPER_NAMES.update({k: k for k in STEREO})     # stereo names already match
PAPER_NAMES.update({k: k for k in CLASSIC})


def per_frame_absrel(key):
    df = pd.read_csv(os.path.join(METRICS, f"{key}_per_image.csv"))
    return df.sort_values("image_id")["all_AbsRel"].to_numpy()


CURVES = {name: per_frame_absrel(key) for name, key in ALL_MODELS.items()}

# ----------------------------------------------------------- AbsRel spread --
# Table II already gives the mean AbsRel and the mean inference time. What no
# table can show is the distribution behind the mean, and that is what this
# figure is for, so it carries accuracy only. It matters because the monocular
# models are strongly right-skewed: their mean sits well above their median,
# and a symmetric error bar would imply error mass that does not exist. Time
# is not drawn -- its spread is small and Table II carries the value.
ROWS = []
for name, key in ALL_MODELS.items():
    kind = "stereo" if name in STEREO else "classic" if name in CLASSIC else "mono"
    ROWS.append((PAPER_NAMES[name], kind, CURVES[name]))
ROWS.sort(key=lambda r: r[2].mean())              # as ordered in Table II

STYLE = {"stereo": C_STEREO, "mono": C_MONO, "classic": C_CLASSIC}

fig, ax = plt.subplots(figsize=(3.4, 2.1))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.25, linewidth=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

bp = ax.boxplot([r[2] for r in ROWS], positions=list(range(len(ROWS))),
                widths=0.62, whis=(5, 95), showfliers=False,
                patch_artist=True, showmeans=True,
                meanprops=dict(marker="D", markersize=2.4,
                               markerfacecolor="white",
                               markeredgecolor="0.15", markeredgewidth=0.5),
                zorder=3)
for n, (label, kind, curve) in enumerate(ROWS):
    color = STYLE[kind]
    bp["boxes"][n].set(facecolor=color, edgecolor=color, linewidth=0.6, zorder=3)
    bp["medians"][n].set(color="white", linewidth=0.9, zorder=4)
    for art in (bp["whiskers"][2 * n], bp["whiskers"][2 * n + 1],
                bp["caps"][2 * n], bp["caps"][2 * n + 1]):
        art.set(color=color, linewidth=0.8, zorder=3)

ax.set_xticks(range(len(ROWS)), [r[0] for r in ROWS],
              rotation=35, ha="right", rotation_mode="anchor")
ax.set_xlim(-0.7, len(ROWS) - 0.3)
ax.set_ylabel("AbsRel")
ax.set_ylim(0, 0.45)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])

handles = [plt.Line2D([], [], marker="s", color=c, linestyle="none", markersize=4)
           for c in (C_MONO, C_STEREO, C_CLASSIC)]
ax.legend(handles, ["monocular", "stereo", "SGBM"], loc="upper left",
          frameon=False, handletextpad=0.3, labelspacing=0.25,
          borderpad=0.0, borderaxespad=0.2)

fig.savefig(os.path.join(OUT, "absrel_spread.pdf"), bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------- predicted depth maps --
# One shared colour scale across the four panels, so a single large colour bar
# replaces the four unreadable per-panel ones and the scale differences between
# the models become directly visible instead of being normalised away.
PREVIEW_FRAME = 3
PANELS = [("UniDepthV2", "UniDepth_mono"),
          ("DA3", "DepthAnything3_mono"),
          ("MoGe-2", "MoGe_mono"),
          ("BridgeDepth (RVC)", "BridgeDepth_rvc")]
VMIN, VMAX = 0.25, 1.40          # metres; covers the 1st-99th pct of all four
# Fig. 2 is a thesis composite that cannot be re-rendered here; sampling its
# reference-depth panel identifies its colour map as turbo (mean RGB distance
# 0.0035, against 0.19 for jet), so Fig. 3 uses turbo to stay consistent.
CMAP = "turbo"

maps = {}
for label, key in PANELS:
    d = np.load(os.path.join(DATA, "depth_estimation", key, "depth",
                             f"{PREVIEW_FRAME}_depth.npy")).astype(float)
    maps[label] = np.where(np.isfinite(d), d, np.nan)

fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.62))
im = None
for ax, (label, _) in zip(axes, PANELS):
    d = maps[label]
    im = ax.imshow(d, cmap=CMAP, vmin=VMIN, vmax=VMAX, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color("0.4")
    # the per-panel range goes in the title, where it cannot collide with the
    # shared colour bar underneath
    lo, hi = np.nanmin(d), np.nanmax(d)
    ax.set_title(f"{label}\n{lo:.2f}--{hi:.2f} m", fontsize=8, pad=2,
                 linespacing=1.35)

fig.subplots_adjust(left=0.005, right=0.995, top=0.80, bottom=0.26, wspace=0.035)
cax = fig.add_axes([0.29, 0.150, 0.42, 0.062])
# only the upper end is clipped -- every model's minimum is above VMIN
cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max")
cb.solids.set_edgecolor("face")   # kill the white seam in vector output
cb.set_label("predicted depth [m]", fontsize=8, labelpad=2)
cb.set_ticks([0.25, 0.50, 0.75, 1.00, 1.25])
cb.ax.tick_params(labelsize=7.5, length=2, pad=1)
cb.outline.set_linewidth(0.4)
fig.savefig(os.path.join(OUT, "predictions.pdf"), bbox_inches="tight")
plt.close(fig)

print("written:", sorted(os.listdir(OUT)))
