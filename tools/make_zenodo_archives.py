"""Pack the data for the Zenodo record.

Builds the archives described in the README ("Data") from ``<ROOT>/datasets``
and ``<ROOT>/out``, plus a ``SHA256SUMS`` file. Every archive stores paths
relative to ``<ROOT>`` (``datasets/dataset_24042026/...``, ``out/out_24042026/...``),
so unpacking all of them into the repository root gives the layout the code
expects.

Depth maps and images do not compress, so files are stored, not deflated; that
also keeps packing fast. Windows ``*Zone.Identifier`` streams, ``__pycache__``
and the stale backups listed in ``EXCLUDE`` are left out.

    python tools/make_zenodo_archives.py --list                    # sizes only
    python tools/make_zenodo_archives.py --out D:/zenodo_upload    # build all
    python tools/make_zenodo_archives.py --out D:/zenodo_upload --only out_core
"""
import argparse
import fnmatch
import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # runs without `pip install -e .`
from nico_stereo.config import DATASET_DIR, OUT_DIR, ROOT  # noqa: E402

D = DATASET_DIR.relative_to(ROOT).as_posix()
O = OUT_DIR.relative_to(ROOT).as_posix()

ARCHIVES = {
    # key: (file name, [paths relative to ROOT])
    "out_core": ("nico_stereo_out_core.zip", [
        f"{O}/cameras_parameters",
        f"{O}/cameras_statistic_model",
        f"{O}/depth_comparison",
        f"{O}/inference_time_stats.csv",
        f"{O}/pose_estimation/3D_models",
        f"{O}/pose_estimation/masks",
        f"{O}/pose_estimation/results",
        f"{O}/pose_estimation/results_check",
    ]),
    "out_depth_estimation": ("nico_stereo_out_depth_estimation.zip", [
        f"{O}/depth_estimation",
    ]),
    "out_pose_inputs": ("nico_stereo_out_pose_inputs.zip", [
        f"{O}/pose_estimation/{d}" for d in (
            "depth_nn", "depth_rgbd", "depth_sgbm", "depth_las2_m", "depth_las2_h",
            "undistorted_images_NICO")
    ]),
    "dataset_stereo_4k_depth": ("nico_stereo_dataset_stereo_4k_depth.zip", [
        f"{D}/stereo_4k_depth",
    ]),
    "dataset_downstream_task": ("nico_stereo_dataset_downstream_task.zip", [
        f"{D}/downstream_task",
    ]),
    "dataset_calibration": ("nico_stereo_dataset_calibration.zip", [
        f"{D}/{d}" for d in (
            "stereo_4k_calibration", "stereo_4k_relative_pose", "calibration_ZED",
            "calibration_Realsense", "distance_validation")
    ]),
    "dataset_camera_stats_model": ("nico_stereo_dataset_camera_stats_model.zip", [
        f"{D}/camera_stats_model",
    ]),
}

EXCLUDE = [
    "*Zone.Identifier",
    "*/__pycache__/*",
    "*/metrics_cauchy_backup_pre_sgbm/*",   # backup of the pre-SGBM metrics
    "*/results_comparison/*",               # stale export, nothing reads it
]


def files_under(rel: str):
    base = ROOT / rel
    paths = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
    for p in paths:
        arc = p.relative_to(ROOT).as_posix()
        if not any(fnmatch.fnmatch(arc, pat) for pat in EXCLUDE):
            yield p, arc


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None, help="directory for the archives")
    ap.add_argument("--only", nargs="+", choices=sorted(ARCHIVES), help="build just these")
    ap.add_argument("--list", action="store_true", help="print file counts and sizes, write nothing")
    args = ap.parse_args()
    if not args.list and args.out is None:
        ap.error("--out is required unless --list is given")

    sums = []
    for key in args.only or ARCHIVES:
        name, rels = ARCHIVES[key]
        missing = [r for r in rels if not (ROOT / r).exists()]
        if missing:
            sys.exit(f"{name}: missing under {ROOT}: {', '.join(missing)}")
        entries = [e for r in rels for e in files_under(r)]
        total = sum(p.stat().st_size for p, _ in entries)
        print(f"{name:<48} {len(entries):>7} files  {total / 1e9:7.2f} GB")
        if args.list:
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        target = args.out / name
        with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for p, arc in entries:
                z.write(p, arc)
        sums.append(f"{sha256(target)}  {name}")
        print(f"  wrote {target}")

    if sums:
        sums_file = args.out / "SHA256SUMS"
        old = sums_file.read_text().splitlines() if sums_file.exists() else []
        keep = [l for l in old if l.split("  ", 1)[-1] not in {s.split("  ", 1)[1] for s in sums}]
        sums_file.write_text("\n".join(sorted(keep + sums)) + "\n")
        print(f"checksums in {sums_file}")


if __name__ == "__main__":
    main()
