"""Where things live on disk.

Every script resolves its inputs and outputs through this module, so the
repository can be checked out anywhere. By default ``ROOT`` is the repository
root, which is where the data repository is expected to be downloaded::

    <ROOT>/datasets/dataset_24042026/    raw recordings   (HF: kocurvik/nico_stereo_data)
    <ROOT>/out/out_24042026/             evaluation data  (HF: kocurvik/nico_stereo_data)
    <ROOT>/third_party/<repo>/           clones of the models' own repositories

Each location can be overridden with an environment variable, e.g. to keep the
~15 GB of data on another drive:

    NICO_STEREO_ROOT          directory that contains ``datasets/`` and ``out/``
    NICO_STEREO_THIRD_PARTY   directory that contains the model clones
    FOUNDATIONPOSE_DIR        FoundationPose clone (default: <THIRD_PARTY>/FoundationPose)
"""
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]

ROOT = Path(os.environ.get("NICO_STEREO_ROOT") or REPO_DIR)
THIRD_PARTY = Path(os.environ.get("NICO_STEREO_THIRD_PARTY") or REPO_DIR / "third_party")
FOUNDATIONPOSE_DIR = Path(os.environ.get("FOUNDATIONPOSE_DIR") or THIRD_PARTY / "FoundationPose")

DATE = "24042026"
DATASET_DIR = ROOT / "datasets" / f"dataset_{DATE}"
OUT_DIR = ROOT / "out" / f"out_{DATE}"
