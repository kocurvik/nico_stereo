# nico_stereo

Code for **"Built-In Stereo vs. RGB-D: Practical Depth and 6D Object Pose Estimation on the NICO Humanoid"**
(Palider and Kocur, ICETA 2026).

The paper asks whether the built-in wide-angle stereo pair of the humanoid robot NICO can replace a
dedicated RGB-D camera. A calibrated dataset records the same scenes with NICO's stereo pair, an Intel
RealSense D435i and a Stereolabs ZED M. The code in this repository calibrates the three devices,
characterises the RGB-D noise, runs and scores five stereo networks, three monocular networks and a
classical SGBM baseline against the ZED M, and compares FoundationPose 6D object poses obtained from
the different depth sources.

> **Authorship.** The work was done mostly by **Matej Palider** as part of his master's thesis
> *"Extrakcia hĺbkovej mapy pomocou stereokamery robota NICO"* (Comenius University in Bratislava,
> Faculty of Mathematics, Physics and Informatics, 2026; supervisor Viktor Kocur). The thesis code
> base is the starting point of this repository. The `analysis/` and `paper_scripts/` folders, the SGBM and
> LiteAnyStereo V2 runners and the FoundationPose driver `run_pose_arm.py` were written afterwards for
> the paper.

- **Data:** Zenodo, <https://doi.org/10.5281/zenodo.XXXXXXX> (see [Data](#data))
- **License:** none chosen yet.

## Contents

```text
nico_stereo/                  the Python package (imported as `nico_stereo`)
    config.py                 where data and third-party repos live (see "Setup")
    calibration/              camera capture and calibration, ChArUco relative poses, validation
    depth_compute/            runners that live inside this package: SGBM, LiteAnyStereo V2
    depth_compare/            warps predictions to the ZED M view and scores them (Cauchy-aware metrics)
    depth_noise_model/        noise distributions of the two RGB-D cameras, ZED M confidence intervals
    RGBD_task/                6D pose task: masks, FoundationPose driver, pose comparison
    charts/, stats/           thesis plots and timing tables
    image.py, utils.py, prepare_paths.py, rectify_omni_pairs.py, ...   shared helpers
depth_estimation_scripts/     inference scripts of the delivered networks (run inside each model's repo)
downstream_task_scripts/      the original FoundationPose drivers for the ZED M / RealSense arms
analysis/                     scores behind Tables II-IV; results/ holds their CSVs
paper_scripts/                verify_data.py, figures, inference-time measurement
tools/make_zenodo_archives.py packs the data for Zenodo
third_party_patches/          Windows build patch for FoundationPose
```

## Setup

```bash
git clone https://github.com/kocurvik/nico_stereo
cd nico_stereo
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` is enough for everything that only reads the delivered data (scoring, figures,
`verify_data.py`). It was tested with Python 3.10 on Windows 11. `opencv-contrib-python` is required,
because the calibration and rectification use `cv2.omnidir`.

**Where things live.** `nico_stereo/config.py` resolves every path. By default the repository root
holds the data and the third-party clones:

```text
<repo>/datasets/dataset_24042026/     raw recordings            (Zenodo)
<repo>/out/out_24042026/              evaluation data           (Zenodo)
<repo>/third_party/<name>/            clones of the model repositories
```

To keep the ~17 GB elsewhere set `NICO_STEREO_ROOT` (the directory that contains `datasets/` and `out/`),
`NICO_STEREO_THIRD_PARTY` and `FOUNDATIONPOSE_DIR`. The date `24042026` is the recording date and is
hard-coded as a default in several scripts.

**Run everything from the repository root.** Modules run as `python -m nico_stereo.<module>`, scripts as
`python <folder>/<script>.py`.

## Data

The data are archived on Zenodo as one record with seven zip files. Every archive stores paths
relative to the repository root, so unpacking all you need into the repository root gives the layout
above. Download with `wget "https://zenodo.org/records/XXXXXXX/files/<file>?download=1"` or the Zenodo web
page, check them against `SHA256SUMS`, then `unzip` each into the repository root.

| file | size | contents | needed for |
|---|---|---|---|
| `nico_stereo_out_core.zip` | 0.3 GB | `out/out_24042026/`: `cameras_parameters`, `cameras_statistic_model`, `depth_comparison`, `inference_time_stats.csv`, and `pose_estimation/{3D_models,masks,results,results_check}` (the FoundationPose poses of all arms) | every scoring script |
| `nico_stereo_out_depth_estimation.zip` | 6.0 GB | `out/out_24042026/depth_estimation/<model>/`: the depth maps (`depth/*.npy`, 215 frames, 640x360), `run_stats.json`, visualisations for 13 depth sources | depth scoring, figures, `verify_data.py` |
| `nico_stereo_out_pose_inputs.zip` | 0.9 GB | `out/out_24042026/pose_estimation/`: `depth_{nn,rgbd,sgbm,las2_m,las2_h}` and `undistorted_images_NICO` | re-running FoundationPose |
| `nico_stereo_dataset_stereo_4k_depth.zip` | 5.5 GB | `datasets/dataset_24042026/stereo_4k_depth/`: the 215 evaluation frames (4K stereo pair, RealSense and ZED M RGB and depth) | re-running depth networks, re-scoring from raw |
| `nico_stereo_dataset_downstream_task.zip` | 1.5 GB | `.../downstream_task/`: the six pose scenes (scenes 001, 002, 004, 005, 006, 009) | re-running the pose pipeline from raw |
| `nico_stereo_dataset_calibration.zip` | 1.5 GB | `.../stereo_4k_calibration`, `stereo_4k_relative_pose`, `calibration_ZED`, `calibration_Realsense`, `distance_validation` | re-running calibration |
| `nico_stereo_dataset_camera_stats_model.zip` | 1.6 GB | `.../camera_stats_model/`: repeated static scenes behind the noise model | re-running the noise model |

The depth maps are metric depth in metres, one `<frame>_depth.npy` per frame. `depth_comparison/zed/metrics_cauchy/`
holds the per-image and summary metrics, `pose_estimation/results/<arm>/<scene>/<object>/ob_in_cam/<frame>.txt` the
4x4 FoundationPose poses. The arms are `left` (BridgeDepth RVC, delivered), `zed`, `realsense` (delivered),
`sgbm`, `las2_m`, `las2_h` (produced for the paper) and, in `results_check/left`, the validation run of the driver.

To rebuild the archives from a local copy of the data use `python tools/make_zenodo_archives.py --out <dir>`
(`--list` prints only the sizes).

## Reproducing the paper

Each step below needs only the files named in its row of the data table.

### 1. Check every number in the paper against the data

```bash
python paper_scripts/verify_data.py
```

Needs `out_core` and `out_depth_estimation`. Re-derives every number the paper states (Tables I-IV and
the claims in Sections V-VI) from the exports and the CSVs in `analysis/results/`, and exits non-zero on any
disagreement. This repository's last run: 54 checks, all passed.

### 2. Re-score the depth maps (Tables II and III)

```bash
python -m nico_stereo.depth_compare.compute_cauchy_metrics   # per-model metrics on each model's own valid pixels
python analysis/realsense_as_prediction.py                   # RealSense scored as if it were a network
python analysis/common_support_comparison.py                 # all sources on the pixels every source has (control)
python analysis/common_support_las2.py                       # Table II: the same with LAS2-M/H included
python analysis/eval_liteanystereo.py LiteAnyStereoV2_M LiteAnyStereoV2_H --control S2M2_stereo
```

`compute_cauchy_metrics` rewrites `out/.../depth_comparison/zed/metrics_cauchy/` (copy it first if you want
to keep the delivered files). The scripts in `analysis/` write their CSVs to `analysis/results/`. Every model
is compared at 640x360 against the ZED M depth warped into the same view, with the validity-weighted resize
used throughout. Table II is scored on one common pixel set (ZED valid, RealSense valid and every method
valid, 76,249 px per frame), so adding or removing a model changes every row.

`eval_liteanystereo.py` scores sources that are not in `DEPTH_ESTIMATION_NETWORKS`
(`nico_stereo/prepare_paths.py`) without changing the published numbers; `--control` re-scores a delivered model
as a check that the script reproduces `all_networks_summary.csv`.

### 3. Re-score the 6D poses (Table IV)

```bash
python analysis/pose_adds_comparison.py     # ZED M / RealSense / network / SGBM pairings, ADD-S and centroid distance
python analysis/pose_adds_las2.py           # the LAS2-M and LAS2-H arms, beside the pairings above
python analysis/validate_driver.py          # does run_pose_arm.py reproduce the delivered `left` arm?
```

Needs `out_core` only. These read the saved poses and the CAD models, and estimate nothing. Poses are compared
with **ADD-S** (mean nearest-neighbour distance between the two placed model point clouds), never with the raw
translation: the CAD origins sit up to 14 cm from the object centroids and the objects are symmetric, so the
translation of two equally good poses can differ by 20 cm.
`nico_stereo/RGBD_task/compare_6D_pose.py` provides the pose and calibration helpers they share. Its rotation
column is deliberately not used: it applies the same ten rotations to every object, which is wrong for the
asymmetric ones.

### 4. Redraw the figures

```bash
python paper_scripts/make_figures.py        # Fig. 3 (predictions.pdf) and the AbsRel spread
python paper_scripts/make_pose_figure.py    # Fig. 4 panels, from FoundationPose track_vis renders
python paper_scripts/make_noise_figure.py   # noise histograms (slow: ~40 M samples; not in the paper)
```

Outputs go to `paper_scripts/figures/`. `make_figures.py` needs a LaTeX installation on `PATH`: it sets
`text.usetex` so the labels use the paper's font. Fig. 1 and Fig. 2 are photographs/composites from the thesis
and are not regenerated here.

### 5. Re-run the depth models

The nine delivered models were run on the 215 frames by the scripts in `depth_estimation_scripts/`, each in
the environment of the model's own repository. SGBM and LiteAnyStereo V2 were added later and run from this package.

| model | repository (commit used) | script |
|---|---|---|
| FoundationStereo | [NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo) (`6e88068`) | `run_inference_foundation_stereo.py` |
| S2M2 | [junhong-3dv/s2m2](https://github.com/junhong-3dv/s2m2) (`8f3bd78`) | `run_inference_s2m2.py` |
| BridgeDepth | [aeolusguan/BridgeDepth](https://github.com/aeolusguan/BridgeDepth) (`9a23160`) | `run_inference_bridge_depth.py` |
| DEFOM-Stereo | [insta360-research-team/defom-stereo](https://github.com/insta360-research-team/defom-stereo) (`5b27591`) | `run_inference_DEFOM.py` |
| UniDepthV2 | [lpiccinelli-eth/UniDepth](https://github.com/lpiccinelli-eth/UniDepth) (`8d8cfe4`) | `run_inference_UniDepth.py` |
| MoGe-2 | [microsoft/moge](https://github.com/microsoft/moge) | `run_inference_moge.py` |
| Depth Anything 3 | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (`3d835ec`) | `run_inference_depth_anything.py` |
| LiteAnyStereo V2 | [TomTomTommi/LiteAnyStereo](https://github.com/TomTomTommi/LiteAnyStereo) (`8c97bd4`) | `python -m nico_stereo.depth_compute.run_liteanystereo --model_size m` |
| SGBM (OpenCV) | none | `python -m nico_stereo.depth_compute.run_sgbm` |

Clone the repositories into `third_party/` and install each as its authors describe. Install this package
into the same environment (`pip install -e <this repo>`), because the scripts import `nico_stereo`. Run the
script from inside that environment; the scripts take the checkpoint locations from their position in the
model's repository (see the `--checkpoint_path` / `--pretrain_dir` defaults), so either copy the script into a
subfolder of the clone or pass the path explicitly. Each writes
`out/out_24042026/depth_estimation/<RunName>/{depth,vis}/` and `run_stats.json`, which `compute_cauchy_metrics`
(step 2) then picks up. The scripts share `--date`, `--scale` (6, i.e. 640x360) and `--max_imgs`; see `--help`.
Checkpoints are listed in each model's own README. LiteAnyStereo V2 additionally needs `timm>=1.0` for
`fasternet_t0`; the runner expects it in `third_party/LiteAnyStereo/_vendor`.

Three conventions are easy to break and then give plausible but wrong numbers:

- **SGBM parameters are fixed a priori and were not tuned.** `numDisparities = 64` follows from the geometry
  (f*B = 14.69 px*m at 640x360, 0.25 m is 59 px of disparity); the rest is the OpenCV reference configuration. The
  networks are used off the shelf, so tuning the baseline on the 215 evaluation frames would make it
  incomparable.
- **Rectification must use `new_K_l`, never the wide intrinsics.** Do not pass `--use-wide` to
  `rectify_omni_pairs.py` for anything on this path: the comparison reads `new_K_l` as the source intrinsics, so
  the wide variant produces depth maps that look fine and warp wrongly.
- **Images are fed to every network as BGR**, as delivered (`image.py` has the BGR to RGB conversion commented
  out). For LAS2 this makes no practical difference (`--rgb` gives the same AbsRel); it was not tested for the
  other networks.

**Inference time** (Table II) is not taken from `run_stats.json`, which belongs to the original runs on a machine
that is no longer available. It was re-measured on one machine by

```bash
python paper_scripts/export_frames.py all           # rectify and downscale once
python paper_scripts/retime.py <model>              # sgbm | unidepth | da3 | moge | bridgedepth_rvc | bridgedepth_mid
                                                    # | foundationstereo | s2m2 | defom | las2_m | las2_h
```

and stored in `paper_scripts/retime_results.json` and `retime_las2.json` (30 evenly spaced frames, three passes,
Intel Core i7-11800H with an NVIDIA RTX A4000 Laptop GPU). `retime.py` drives each model through the timed
region of its own inference script and expects the model repositories in `third_party/`.

### 6. Re-run the 6D pose estimation

FoundationPose ([NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose), commit `a1b694b`) is used in
its model-based mode with the CAD meshes in `pose_estimation/3D_models`. Install it as its authors describe into
`third_party/FoundationPose` (or point `FOUNDATIONPOSE_DIR` at an existing clone). On Windows the unmodified
repository does not build; `third_party_patches/FoundationPose-windows.patch` fixes the `mycpp` extension (MSVC,
no Boost, no `unistd.h`) and the hard-coded `/tmp` path, and does not change the estimator:

```bash
cd third_party/FoundationPose && git apply ../../third_party_patches/FoundationPose-windows.patch
```

Depth for the six pose scenes comes from each depth source, converted to 16-bit PNG in millimetres in
`pose_estimation/depth_<arm>/<scene>/depth_png/`. The delivered files are in `out_pose_inputs`; to regenerate:

```bash
python -m nico_stereo.depth_compute.run_sgbm_scenes
python -m nico_stereo.depth_compute.run_liteanystereo_scenes --model_size m     # and h
python depth_estimation_scripts/run_inference_bridge_depth_scenes.py            # network arm, inside the BridgeDepth env
python -m nico_stereo.RGBD_task.depth_images_scenes                             # .npy -> depth_png
```

Then run the driver in an environment where FoundationPose imports (CUDA, `nvdiffrast`, `mycpp` built):

```bash
python -m nico_stereo.RGBD_task.run_pose_arm --arm sgbm        # nn | sgbm | las2_m | las2_h
python -m nico_stereo.RGBD_task.run_pose_arm --arm nn --out results_check/left     # validation run
```

and score as in step 3. `run_pose_arm.py` writes `results/<arm>/` next to the delivered arms (`--out` changes
that). Details that the driver relies on, and that cost time to find:

- The working resolution is `shorter_side=324` (576x324), with the intrinsics `new_K_l` scaled to it.
- **Frames are indexed by integer, never by sorted filename.** The depth files are `0.png ... 10.png`, which sort
  as `0, 1, 10, 2, ...`; FoundationPose's own `YcbineoatReader` sorts that way and would pair frame 1 with frame
  10.
- The three delivered arms (`left`, `zed`, `realsense`) were produced on the original author's Linux machine by
  `downstream_task_scripts/run_inference_{NICO,RGBD}.py`, which are meant to be placed in the FoundationPose
  clone. `run_pose_arm.py` is a reconstruction of that driver for the robot-head arms. Re-running the network
  depth through it reproduces the delivered `left` arm to a median ADD-S of 0.17 cm over 376 frames
  (`analysis/validate_driver.py`), so all arms are measured the same way.
- `mask_creation_iterative.py` is the interactive tool that drew the object masks in `pose_estimation/masks/`.
  The masks are delivered; nothing needs to be redrawn.
- `render_track_vis.py` redraws FoundationPose's `track_vis` overlay from saved poses (used for Fig. 4).

### 7. From the raw recordings

Calibration and the noise model are the thesis-time stages that produced `out/.../cameras_parameters` and
`cameras_statistic_model`; both are delivered, so nothing above needs them. In order:

1. `python -m nico_stereo.calibration.calibrate_stereo <calib_imgs_dir> <out_dir>`: omnidirectional
   (`cv2.omnidir.stereoCalibrate`) calibration of the two head cameras from ChArUco boards
   (`calibration/ChArUco/`, board images A3 5x7 / 6x8), written to `calib_data.npy`. Images are in
   `stereo_4k_calibration/`.
2. `python -m nico_stereo.calibration.calibrate`: pinhole calibration of the RealSense and the ZED M from the
   chessboard images in `calibration_Realsense/` and `calibration_ZED/` (8x5 board, 30 mm squares), written to
   `cameras_parameters/{realsense,zed}_calibration_1280x720.yaml`.
   (`RGBD_cameras_instrinct_parameters.py` instead reads the *factory* intrinsics from connected devices.)
3. `python -m nico_stereo.calibration.estimate_relative_pose`: relative poses between the head cameras and the
   RGB-D cameras from synchronous ChArUco views in `stereo_4k_relative_pose/`
   (`relative_pose_*_to_left_v3.yaml`); `validate.py` and `validation_point_distance.py` check the result by
   point distances on `distance_validation/`.
4. `python -m nico_stereo.depth_noise_model.stats_model`: fits the error distributions of both RGB-D cameras on
   `camera_stats_model/` and writes `best_distribution_models.json` (RealSense: Student's t, ZED M: Cauchy). This is
   a script, not a function: importing it runs it. `CI_calculation.py` turns the ZED M scale into the confidence
   intervals used by the coverage metrics.
5. `nico_stereo/rectify_omni_pairs.py`, `visualize_depth.py`, `visualize_point_transfer.py`: rectification and
   inspection helpers. Run `--help` on any of these for their options; none was re-run for this release.

`nico_stereo/calibration/get_images.py` and `capture_realsense.py` capture data and need the vendor SDKs
(`pyrealsense2`, `pyzed`), which are not in `requirements.txt`.

## What was checked in this release

The repository was cleaned up for publication without changing any computation: the package was renamed
from `code` (which shadows the standard library), all paths now go through `nico_stereo/config.py`, and
unused scripts, copyrighted example images and machine-specific files were removed. What was run:

- **Run, on the delivered data:** `paper_scripts/verify_data.py` (54 checks, all passed);
  `analysis/pose_adds_comparison.py` and `analysis/pose_adds_las2.py` (their four CSVs came out byte-identical to
  the delivered ones); `tools/make_zenodo_archives.py` (built and read back `nico_stereo_out_core.zip`).
- **Checked statically only:** every `nico_stereo.*` import resolves and every file compiles. The other
  scripts (depth-model inference, FoundationPose driver, calibration, figures, timing) were not executed again
  after the cleanup, and the depth maps and poses were not regenerated. The delivered outputs are the
  paper's results.
- Some comments and console messages in the thesis-time modules (`depth_noise_model/stats_model.py`, parts of
  `calibration/`) are in Slovak.
