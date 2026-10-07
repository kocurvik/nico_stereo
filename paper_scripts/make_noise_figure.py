"""Redraw the depth-noise histograms of the two RGB-D cameras in English.

This is a port of ``nico_stereo/depth_noise_model/stats_model.py``
into the visual style of the other paper figures (matplotlib, serif, 8 pt,
vector PDF).  The epsilon computation, the edge mask and the distribution
fitting are kept identical to the original, so the KS statistics it prints
must reproduce ``out/out_24042026/cameras_statistic_model/
best_distribution_models.json``.

It is kept separate from ``make_figures.py`` because it reads ~1.5 GB of raw
``.npy`` depth stacks and takes a few minutes.  The resulting
``figures/noise_model.pdf`` is currently spare material -- ``main.tex`` does not
reference it, because Table I already carries the numbers that matter and the
paper has no room for another figure.

Run:  python paper_scripts/make_noise_figure.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.ndimage import generic_gradient_magnitude, sobel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # lets the script run without `pip install -e .`
from nico_stereo.config import ROOT  # noqa: E402
OUT = os.path.join(HERE, "figures")
DATE = "24042026"
BASE = os.path.join(ROOT, "datasets", f"dataset_{DATE}", "camera_stats_model")
REF = os.path.join(ROOT, "out", f"out_{DATE}", "cameras_statistic_model",
                   "best_distribution_models.json")

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

DISTRIBUTIONS = {"norm": stats.norm, "laplace": stats.laplace,
                 "cauchy": stats.cauchy, "t": stats.t,
                 "logistic": stats.logistic}
PRETTY = {"norm": "normal", "laplace": "Laplace", "cauchy": "Cauchy",
          "t": "Student's $t$", "logistic": "logistic"}
CAMERAS = {"realsense": ("RealSense D435i", "#e67e22", "#8a4b12"),
           "zed": ("ZED M", "#1f4e9c", "#12305e")}


def compute_edge_mask(z_ref, threshold=0.05):
    """True where the reference depth is locally planar (no flying pixels)."""
    z_filled = np.where(np.isfinite(z_ref), z_ref, 0.0)
    grad = generic_gradient_magnitude(z_filled, sobel).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_grad = grad / np.where(z_filled > 0, z_filled, np.nan)
    return (rel_grad < threshold) & np.isfinite(z_ref)


def collect_eps(camera):
    scenes = sorted(d.path for d in os.scandir(BASE) if d.is_dir())
    chunks = []
    for scene in scenes:
        depth_dir = os.path.join(scene, "depth")
        files = sorted(f.path for f in os.scandir(depth_dir)
                       if f.name.endswith(f"{camera}_depth.npy"))
        if not files:
            continue
        stack = np.stack([np.load(f).astype(np.float32) for f in files], axis=0)
        stack[stack <= 0] = np.nan

        z_ref = np.nanmean(stack, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel_err = (stack - z_ref[np.newaxis]) / z_ref[np.newaxis]

        mask3d = np.broadcast_to(compute_edge_mask(z_ref)[np.newaxis],
                                 rel_err.shape)
        eps = rel_err[mask3d].ravel()
        chunks.append(eps[np.isfinite(eps)])
    return np.concatenate(chunks)


def best_fit(eps):
    results = {}
    for name, dist in DISTRIBUTIONS.items():
        params = dist.fit(eps, floc=0.0)
        ks_stat, _ = stats.kstest(eps, name, args=params)
        results[name] = (params, ks_stat)
    return min(results.items(), key=lambda kv: kv[1][1])


reference = json.load(open(REF)) if os.path.exists(REF) else {}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
for ax, (camera, (title, fill, line)) in zip(axes, CAMERAS.items()):
    eps = collect_eps(camera)
    name, (params, ks_stat) = best_fit(eps)
    print(f"[{camera}] n={len(eps):,}  best={name}  KS={ks_stat:.4f}  "
          f"params={params}")
    if camera in reference:
        r = reference[camera]
        assert r["best_distribution"] == name, f"{camera}: distribution differs"
        assert abs(r["ks_stat"] - ks_stat) < 1e-6, f"{camera}: KS differs"
        print(f"           reproduces best_distribution_models.json")

    x_min, x_max = -0.04, 0.04
    clipped = eps[(eps >= x_min) & (eps <= x_max)]
    counts, edges = np.histogram(clipped, bins=120, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    x_g = np.linspace(x_min, x_max, 400)

    ax.bar(centers, counts, width=edges[1] - edges[0], color=fill, alpha=0.70,
           linewidth=0, label=r"histogram of $\varepsilon$")
    ax.plot(x_g, DISTRIBUTIONS[name].pdf(x_g, *params), color=line,
            linewidth=1.4, label=f"fit: {PRETTY[name]}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"relative error $\varepsilon$")
    ax.set_title(f"{title} (KS $={ks_stat:.4f}$)", pad=4)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3, linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("probability density")
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUT, "noise_model.pdf"), bbox_inches="tight")
plt.close(fig)
print("written:", os.path.join(OUT, "noise_model.pdf"))
