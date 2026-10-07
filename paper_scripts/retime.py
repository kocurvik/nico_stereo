"""Re-measure inference time for every model on one machine.

The Time column of Table II and the timings quoted in Section V-C come from
here. Each model is driven through exactly the timed region of its delivered
run script in `depth_estimation_scripts/` (and, for SGBM,
`nico_stereo/depth_compute/run_sgbm.py`): the same warm-up count, the same explicit
`cuda.synchronize()` calls or lack of them, and tensor construction outside
the timer, as there.

Only timing is re-measured. The predicted depth maps under
`out/out_24042026/depth_estimation/` are the delivered ones and are not
touched, so every accuracy number in the paper is unaffected.

Inputs come from `export_frames.py`, which rectifies and downscales once --
neither step is inside any timed region.

    python export_frames.py all
    python retime.py <model> [--frames all] [--out paper_scripts/retime_results.json]

    sgbm | unidepth | da3 | moge
    bridgedepth_rvc | bridgedepth_mid | foundationstereo | s2m2 | defom

Results accumulate per model into the --out JSON: per-image times plus the
mean/std the paper quotes.

The four learned stereo repos live under STEREO_NETS, cloned from upstream,
with the checkpoints the paper used. UniDepth and Depth-Anything-3 are
pip-installed into the same env.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # lets the script run without `pip install -e .`
from nico_stereo.config import THIRD_PARTY  # noqa: E402

STEREO_NETS = THIRD_PARTY

RUN_NAME = {
    "sgbm": "SGBM_stereo", "unidepth": "UniDepth_mono",
    "da3": "DepthAnything3_mono", "moge": "MoGe_mono",
    "bridgedepth_rvc": "BridgeDepth_rvc",
    "bridgedepth_mid": "BridgeDepth_middlebury",
    "foundationstereo": "FoundationStereo", "s2m2": "S2M2_stereo",
    "defom": "DEFOM_stereo",
    "las2_m": "LiteAnyStereoV2_M", "las2_h": "LiteAnyStereoV2_H",
}
NEEDS_RIGHT = {"sgbm", "bridgedepth_rvc", "bridgedepth_mid",
               "foundationstereo", "s2m2", "defom", "las2_m", "las2_h"}

ap = argparse.ArgumentParser()
ap.add_argument("which", choices=sorted(RUN_NAME))
ap.add_argument("--frames", default="all")
ap.add_argument("--passes", type=int, default=1,
                help="repeat the whole sweep N times and average the means; "
                     "what the paper quotes is --frames 30 --passes 3")
ap.add_argument("--out", default=str(HERE / "retime_results.json"))
args = ap.parse_args()
WHICH = args.which

FRAMES = HERE / ("_frames_all" if args.frames == "all" else "_frames")
meta = json.load(open(FRAMES / "frames.json"))
ids, w, h = meta["ids"], meta["width"], meta["height"]
K = np.array(meta["K"], dtype=np.float64)

lefts = [cv2.imread(str(FRAMES / f"{i}_left.png")) for i in ids]
rights = ([cv2.imread(str(FRAMES / f"{i}_right.png")) for i in ids]
          if WHICH in NEEDS_RIGHT else [None] * len(ids))
assert all(x is not None for x in lefts)
print(f"[{RUN_NAME[WHICH]}] {len(ids)} frames at {w}x{h}", flush=True)


def finish(times, label, peak_gb=None):
    """`times` is one array per pass; the reported mean averages the passes."""
    per_pass = [np.asarray(x) for x in times]
    t = np.concatenate(per_pass)
    pass_means = np.array([x.mean() for x in per_pass])
    out_path = Path(args.out)
    blob = json.load(open(out_path)) if out_path.exists() else {"models": {}}
    blob.setdefault("_comment", [
        "Inference time measured by paper_scripts/retime.py, one machine, one env.",
        "Each model uses the timed region of its delivered run script.",
        "Only timing; the delivered depth predictions are untouched.",
    ])
    blob["models"][RUN_NAME[WHICH]] = {
        "device": label,
        "n_images": len(per_pass[0]),
        "n_passes": len(per_pass),
        "mean_time_s": round(float(pass_means.mean()), 4),
        "std_time_s": round(float(t.std(ddof=1)), 4),
        "min_time_s": round(float(t.min()), 4),
        "max_time_s": round(float(t.max()), 4),
        "peak_vram_gb": round(peak_gb, 2) if peak_gb else None,
        "per_image": {str(i): round(float(x), 5)
                      for i, x in zip(ids, per_pass[0])},
    }
    json.dump(blob, open(out_path, "w"), indent=1)
    for n, x in enumerate(per_pass):
        print(f"  pass {n + 1}: mean {x.mean():.4f}  sd {x.std(ddof=1):.4f}  "
              f"min {x.min():.4f}  max {x.max():.4f}")
    print(f"  mean over passes {pass_means.mean():.4f}"
          + (f"  peak {peak_gb:.2f} GB" if peak_gb else ""))
    print(f"RESULT {WHICH} {pass_means.mean():.5f}")


# --------------------------------------------------------------------- SGBM --
if WHICH == "sgbm":
    BS = 5
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=64, blockSize=BS,
        P1=8 * 3 * BS * BS, P2=32 * 3 * BS * BS,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=32)
    for _ in range(5):                                   # as run_sgbm.py
        matcher.compute(lefts[0], rights[0])
    passes = []
    for _ in range(args.passes):
        t = []
        for a, b in zip(lefts, rights):
            t0 = time.perf_counter()
            matcher.compute(a, b)
            t.append(time.perf_counter() - t0)
        passes.append(t)
    print(f"OpenCV threads: {cv2.getNumThreads()}")
    finish(passes, "CPU")
    sys.exit(0)

import torch
torch.autograd.set_grad_enabled(False)
device = torch.device("cuda")
print("gpu:", torch.cuda.get_device_name(0), flush=True)

# FoundationStereo and DEFOM build their DINOv2 backbone through
# torch.hub.load("facebookresearch/dinov2", ...), which reaches the network
# even though pretrained_dino=False means no weights are fetched. stdlib SSL
# is broken in this env (ssl.load_default_certs() raises ASN1 NOT_ENOUGH_DATA
# on the Windows cert store), so the call is routed to a local clone -- the
# repos' own comments suggest exactly this.
_HUB_LOCAL = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
if _HUB_LOCAL.is_dir():
    _orig_hub_load = torch.hub.load

    def _hub_load(repo_or_dir, model, *a, **kw):
        if (isinstance(repo_or_dir, str) and "dinov2" in repo_or_dir
                and not Path(repo_or_dir).is_dir()):
            kw["source"] = "local"
            return _orig_hub_load(str(_HUB_LOCAL), model, *a, **kw)
        return _orig_hub_load(repo_or_dir, model, *a, **kw)

    torch.hub.load = _hub_load


def to_nchw(img):
    return torch.as_tensor(img).cuda(non_blocking=True).float()[None].permute(0, 3, 1, 2)


def run_all(prep, call, after, n_warmup, sync):
    """Time `call` per frame; `prep` and `after` stay outside the timer."""
    warm = prep(0)
    for _ in range(n_warmup):
        after(call(warm))
    torch.cuda.synchronize()
    del warm

    passes = []
    for _ in range(args.passes):
        times = []
        for i in range(len(lefts)):
            x = prep(i)
            if sync:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = call(x)
            if sync:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            after(y)                  # forces the sync when there is none
            times.append(dt)
            del x, y
        passes.append(times)
    return passes


# ----------------------------------------------------------------- UniDepth --
if WHICH == "unidepth":
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(device).eval()

    def prep(i):
        return (torch.from_numpy(lefts[i]).permute(2, 0, 1).to(device),
                Pinhole(K=torch.from_numpy(K).unsqueeze(0)))
    times = run_all(prep, lambda x: model.infer(*x),
                    lambda p: p["depth"].squeeze().detach().cpu().numpy(),
                    n_warmup=5, sync=False)
    finish(times, "GPU", torch.cuda.max_memory_reserved() / 1024 ** 3)

# ---------------------------------------------------------------------- DA3 --
elif WHICH == "da3":
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(device)
    model = torch.compile(model, mode="reduce-overhead")
    model.eval()

    def call(x):
        with torch.inference_mode():
            return model.inference([x])
    times = run_all(lambda i: lefts[i], call,
                    lambda p: np.asarray(p.depth[0], dtype=np.float32),
                    n_warmup=5, sync=False)
    finish(times, "GPU", torch.cuda.max_memory_reserved() / 1024 ** 3)

# --------------------------------------------------------------------- MoGe --
elif WHICH == "moge":
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()

    def prep(i):
        return torch.tensor(lefts[i] / 255.0, dtype=torch.float32,
                            device=device).permute(2, 0, 1)
    times = run_all(prep, lambda x: model.infer(x),
                    lambda o: o["depth"].detach().cpu().numpy(),
                    n_warmup=5, sync=False)
    finish(times, "GPU", torch.cuda.max_memory_reserved() / 1024 ** 3)

# -------------------------------------------------------------- BridgeDepth --
elif WHICH.startswith("bridgedepth"):
    sys.path.insert(0, str(STEREO_NETS / "BridgeDepth"))
    from bridgedepth.bridgedepth import BridgeDepth
    name = "rvc" if WHICH.endswith("rvc") else "middlebury"
    ckpt = STEREO_NETS / "BridgeDepth" / "checkpoints" / f"bridge_{name}_pretrain.pth"
    model = BridgeDepth.from_pretrained(str(ckpt)).to(device).eval()

    times = run_all(lambda i: {"img1": to_nchw(lefts[i]), "img2": to_nchw(rights[i])},
                    lambda s: model(s),
                    lambda r: r["disp_pred"].cpu().numpy(),
                    n_warmup=3, sync=True)
    finish(times, "GPU", torch.cuda.max_memory_reserved() / 1024 ** 3)

# --------------------------------------------------------- FoundationStereo --
elif WHICH == "foundationstereo":
    sys.path.insert(0, str(STEREO_NETS / "FoundationStereo"))
    sys.path.insert(0, str(STEREO_NETS / "FoundationStereo" / "core"))
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    ckpt_dir = (STEREO_NETS / "FoundationStereo" / "pretrained_models"
                / "11-33-40" / "model_best_bp2.pth")
    cfg = OmegaConf.load(str(ckpt_dir.parent / "cfg.yaml"))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    for k, v in vars(SimpleNamespace(ckpt_dir=str(ckpt_dir), scale=6, valid_iters=32,
                                     hiera=0, seed=0, max_imgs=None, date="24042026",
                                     min_disp=1e-6, run_name="FoundationStereo")).items():
        cfg[k] = v
    model = FoundationStereo(OmegaConf.create(cfg))
    model.load_state_dict(torch.load(str(ckpt_dir), weights_only=False)["model"])
    model.cuda().eval()

    def prep(i):
        x, y = to_nchw(lefts[i]), to_nchw(rights[i])
        return InputPadder(x.shape, divis_by=32, force_square=False).pad(x, y)

    def call(p):
        with torch.cuda.amp.autocast(True):
            return model.forward(*p, iters=32, test_mode=True)
    times = run_all(prep, call, lambda d: d.float().cpu().numpy(),
                    n_warmup=5, sync=True)
    finish(times, "GPU", torch.cuda.max_memory_reserved() / 1024 ** 3)

# --------------------------------------------------------------------- S2M2 --
elif WHICH == "s2m2":
    sys.path.insert(0, str(STEREO_NETS / "s2m2"))
    sys.path.insert(0, str(STEREO_NETS / "s2m2" / "src"))
    from s2m2.core.model.s2m2 import S2M2
    from s2m2.core.utils.image_utils import image_pad

    model = S2M2(feature_channels=256, dim_expansion=1, num_transformer=3,
                 use_positivity=True, refine_iter=3)
    ck = torch.load(str(STEREO_NETS / "s2m2" / "weights" / "pretrain_weights"
                        / "CH256NTR3.pth"), weights_only=False)
    model.my_load_state_dict(ck.get("state_dict", ck))
    model.cuda().eval()

    def prep(i):
        a = torch.as_tensor(lefts[i]).cuda().float().permute(2, 0, 1).unsqueeze(0)
        b = torch.as_tensor(rights[i]).cuda().float().permute(2, 0, 1).unsqueeze(0)
        return image_pad(a, 32), image_pad(b, 32)

    def call(p):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            return model(*p)
    times = run_all(prep, call, lambda r: r[0].squeeze().float().cpu().numpy(),
                    n_warmup=5, sync=False)
    finish(times, "GPU", torch.cuda.max_memory_reserved() / 1024 ** 3)

# -------------------------------------------------------------------- DEFOM --
elif WHICH == "defom":
    sys.path.insert(0, str(STEREO_NETS / "defom-stereo"))
    sys.path.insert(0, str(STEREO_NETS / "defom-stereo" / "core"))
    from core.defom_stereo import DEFOMStereo
    from core.utils.utils import InputPadder

    margs = SimpleNamespace(
        restore_ckpt=str(STEREO_NETS / "defom-stereo" / "models"
                         / "defomstereo_vitl_sceneflow.pth"),
        dinov2_encoder="vitl", idepth_scale=0.5, hidden_dims=[128, 128, 128],
        corr_implementation="reg", shared_backbone=False, corr_levels=2,
        corr_radius=4, scale_list=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        scale_corr_radius=2, n_downsample=2, context_norm="batch",
        n_gru_layers=3, mixed_precision=False, valid_iters=16, scale_iters=4)
    model = DEFOMStereo(margs)
    ck = torch.load(margs.restore_ckpt, weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.cuda().eval()

    def prep(i):
        x, y = to_nchw(lefts[i]), to_nchw(rights[i])
        return InputPadder(x.shape, divis_by=32).pad(x, y)
    times = run_all(prep, lambda p: model(*p, iters=16, scale_iters=4, test_mode=True),
                    lambda d: d.float().cpu().numpy(), n_warmup=5, sync=False)
    finish(times, "GPU", torch.cuda.max_memory_reserved() / 1024 ** 3)

# --------------------------------------------------------------------- LAS2 --
# Timed region of nico_stereo/depth_compute/run_liteanystereo.py (BGR input, as there).
# LAS2 needs a newer timm than the env has; it lives in the repo's _vendor dir.
elif WHICH.startswith("las2"):
    las = STEREO_NETS / "LiteAnyStereo"
    sys.path.insert(0, str(las))
    sys.path.insert(0, str(las / "_vendor"))
    from core.models import build_model, load_model_weights
    from core.utils.utils import InputPadder

    size = WHICH[-1]
    model = build_model("las2", fnet_pretrained=False, model_size=size, max_disp=192)
    load_model_weights(model, torch.load(str(las / "checkpoints" / f"LAS2_{size.upper()}.pth"),
                                         map_location="cuda"), strict=True)
    model.cuda().eval()

    def prep(i):
        x, y = to_nchw(lefts[i]), to_nchw(rights[i])
        return InputPadder(x.shape, divis_by=32).pad(x, y)
    times = run_all(prep, lambda p: model(*p, max_disp=192, test_mode=True),
                    lambda d: d.float().cpu().numpy(), n_warmup=5, sync=True)
    finish(times, "GPU", torch.cuda.max_memory_reserved() / 1024 ** 3)
