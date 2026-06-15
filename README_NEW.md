# Open-Vocabulary 6D Pose Estimation via LLM + YOLOE + Any6D

<p align="center">
  <img src="assets/final.png" alt="Pipeline Architecture — From Human Language and Vision to 6D Pose" width="900"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-1e64b4?logo=python&logoColor=white&style=flat-square" alt="Python"/>
  <img src="https://img.shields.io/badge/Docker-CUDA_12.1-1e64b4?logo=docker&logoColor=white&style=flat-square" alt="Docker"/>
  <img src="https://img.shields.io/badge/Ollama-Mistral-1e64b4?logo=ollama&logoColor=white&style=flat-square" alt="Ollama"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-1e64b4?logo=opencv&logoColor=white&style=flat-square" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/PyTorch-2.4-1e64b4?logo=pytorch&logoColor=white&style=flat-square" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/YOLOE-open--vocab-1e64b4?logo=yolo&logoColor=white&style=flat-square" alt="YOLOE"/>
  <img src="https://img.shields.io/badge/Any6D-CVPR_2025-1e64b4?logoColor=white&style=flat-square" alt="Any6D"/>
</p>

A full pipeline that estimates the 6D pose of objects described in **natural language**, without requiring object-specific training. A user says *"hand me the mustard bottle"* and the system returns a 4×4 SE(3) transformation matrix ready for robot manipulation.

**Single image inference:**
<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">docker exec any6d_active \
  /opt/conda/envs/Any6D/bin/python3 \
  /workspace/pipeline_scripts/main_pipeline.py \
  --image       /path/to/rgb.png \
  --depth       /path/to/depth.png \
  --mesh        /path/to/object.ply \
  --K           572.4 572.4 325.2 242.0 \
  --instruction "Hand me the mustard bottle"</pre>

---

## Table of Contents

0. [Quick Start](#0-quick-start)
1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Repository Layout](#3-repository-layout)
4. [Environment Setup](#4-environment-setup)
5. [Dataset Setup](#5-dataset-setup)
6. [Core Module Reference](#6-core-module-reference)
7. [Pipeline Scripts](#7-pipeline-scripts)
8. [Running Evaluations](#8-running-evaluations)
9. [Single-Image Inference](#9-single-image-inference)
10. [Results & Metrics](#10-results--metrics)
11. [Unit Tests](#11-unit-tests)
12. [LLM Keyword Calibration](#12-llm-keyword-calibration)
13. [Design Decisions & Known Limitations](#13-design-decisions--known-limitations)
14. [Reproducing Results](#14-reproducing-results)

---

## 0. Quick Start

> **Prerequisites:** NVIDIA GPU, Docker + NVIDIA Container Toolkit, Ollama running on the host.

### Step 1 — Clone the repository

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">git clone https://github.com/&lt;your-org&gt;/open-vocabulary-6d-pose-yoloe.git
cd open-vocabulary-6d-pose-yoloe</pre>

### Step 2 — Build and start the Docker container

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em"># Build the Docker image (first time only, ~30-60 min)
bash build_any6d.sh

# Start the container (GPU passthrough, volume mounts)
docker compose -f Any6D/docker-compose.yml up -d

# Verify the container is running
docker ps | grep any6d_active</pre>

### Step 3 — Pull the LLM model (on the host)

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em"># Install Ollama if not already installed
curl -fsSL https://ollama.com/install.sh | sh

# Pull Mistral (used for keyword extraction)
ollama pull mistral:latest

# The container reaches Ollama at http://172.18.0.1:11434</pre>

### Step 4 — Run the unit tests (no GPU, no dataset needed)

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">docker exec any6d_active \
  /opt/conda/envs/Any6D/bin/python3 \
  /workspace/pipeline_scripts/test_pipeline_functions.py -v</pre>

Expected output: **42 tests, 0 failures, 0 errors.**

### Step 5 — Run an evaluation

**LINEMOD — quick test (object 8 = driller, 3 frames):**

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">docker exec any6d_active \
  /opt/conda/envs/Any6D/bin/python3 \
  /workspace/pipeline_scripts/run_full_pipeline_linemod.py \
  --obj_ids 8 --max_frames 3</pre>

**LINEMOD — full evaluation (all 15 objects, 200 frames each):**

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">docker exec any6d_active \
  /opt/conda/envs/Any6D/bin/python3 \
  /workspace/pipeline_scripts/run_full_pipeline_linemod.py</pre>

**YCB-Video — quick test (object 5 = mustard bottle, scene 52, 3 frames):**

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">docker exec any6d_active \
  /opt/conda/envs/Any6D/bin/python3 \
  /workspace/pipeline_scripts/run_full_pipeline_ycbv.py \
  --obj_ids 5 --scene_id 52 --max_frames 3</pre>

**YCB-Video — full evaluation (all 21 objects):**

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">docker exec any6d_active \
  /opt/conda/envs/Any6D/bin/python3 \
  /workspace/pipeline_scripts/run_full_pipeline_ycbv.py</pre>

**HO3D — quick test (sequence MPM10, 5 frames):**

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">docker exec any6d_active \
  /opt/conda/envs/Any6D/bin/python3 \
  /workspace/pipeline_scripts/run_full_pipeline_ho3d.py \
  --sequences MPM10 --max_frames 5</pre>

**HO3D — full evaluation (all 13 sequences):**

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">docker exec any6d_active \
  /opt/conda/envs/Any6D/bin/python3 \
  /workspace/pipeline_scripts/run_full_pipeline_ho3d.py</pre>

Results (JSON + XLSX) are written to `/workspace/results/` inside the container,
which maps to `Any6D/results/` on the host.

---

## 1. Overview

### What this project does

Given an **RGB-D image** and a **free-form natural language instruction** (e.g. *"grab the cylindrical steel can on the table"*), the pipeline:

1. Sends the instruction to a local **LLM** (Mistral via Ollama) which extracts a concise visual keyword (e.g. `"steel can"`).
2. Passes that keyword to **YOLOE** — an open-vocabulary segmentation model — to obtain a pixel mask of the target object.
3. Feeds the RGB-D image and mask to **Any6D** — a model-free 6D pose estimator — which returns a 4×4 pose matrix relative to a single anchor view.

The output is a **6D pose** (rotation + translation) expressed in the camera frame, suitable for robot grasping.

### Why it matters

Classical pose estimators require either class-specific training or a known 3D CAD model per object. This pipeline requires:
- **No training** on target objects.
- **No CAD model** at inference time (only a single RGB-D anchor image and mesh for registration).
- **Natural language** as the only user interface.

### Supported datasets

| Dataset | Objects | Metrics computed |
|---|---|---|
| LINEMOD (LM) | 15 tabletop objects | ADD, ADD-S |
| YCB-Video (YCBV) | 21 household objects | ADD, ADD-S |
| HO3D | 13 hand-object sequences | ADD-S, AR, MSSD, MSPD, VSD |

---

## 2. System Architecture

```
User instruction (natural language)
          │
          ▼
┌─────────────────────┐
│   LLM (Mistral)     │  Ollama REST API · host 172.18.0.1:11434
│   core/llm.py       │  Extracts 1–3 word visual keyword
└─────────┬───────────┘
          │  keyword (e.g. "yellow mustard bottle")
          ▼
┌─────────────────────┐
│  YOLOE Segmentation │  Open-vocabulary instance segmentation
│  core/detection.py  │  yoloe-26l-seg.pt · conf threshold 0.1
└─────────┬───────────┘
          │  binary pixel mask
          ▼
┌─────────────────────┐
│  Any6D Registration │  Model-free 6D pose estimator (CVPR 2025)
│  estimater.py       │  est.register(K, rgb, depth, ob_mask, iter=5)
└─────────┬───────────┘
          │
          ▼
   4×4 SE(3) pose matrix  →  result.json · pose_4x4.txt · viz_pipeline.png
```

### Anchor correction (HO3D)

For the HO3D dataset, a reference *anchor* frame (with known ground-truth pose) is used to correct pose drift:

```
corrected_pose = (pred_query @ inv(pred_anchor)) @ gt_anchor
```

This formula cancels the systematic error that accumulates when the query and anchor views differ significantly.

---

## 3. Repository Layout

```
open-vocabulary-6d-pose-yoloe/
│
├── README.md                    # Original project README (do not modify)
├── README_NEW.md                # This file — full technical documentation
├── setup_any6d.sh               # One-shot setup: clone, weights, Docker build
├── build_any6d.sh               # Build Docker image only
├── setup_master_env.sh          # Host Python environment (YOLOE + Jupyter)
├── master_vm.sh                 # GCP VM creation (g2-standard-4, NVIDIA L4)
│
└── Any6D/                       # Main implementation directory
    │
    ├── Dockerfile               # CUDA 12.1 + conda + all dependencies
    ├── docker-compose.yml       # GPU passthrough, volume mounts, shm 16 GB
    ├── estimater.py             # Any6D pose estimator class  ← DO NOT EDIT
    ├── metrics.py               # BOP metric primitives       ← DO NOT EDIT
    ├── renderer_pyrender.py     # PyRender-based depth renderer
    │
    ├── core/                    # ★ Shared utilities (DRY module)
    │   ├── __init__.py          # Public API exports
    │   ├── constants.py         # Global constants (paths, thresholds)
    │   ├── llm.py               # Ollama LLM integration
    │   ├── detection.py         # YOLOE singleton + IoU helpers
    │   ├── pose_utils.py        # Rotation/translation error, anchor correction
    │   ├── metrics_utils.py     # ADD, ADD-S, AR, MSSD, MSPD, nanmean
    │   └── io_utils.py          # JSON / XLSX results export
    │
    ├── pipeline_scripts/        # ★ End-to-end evaluation pipelines
    │   ├── main_pipeline.py                  # Single-image full pipeline
    │   ├── run_full_pipeline_linemod.py      # LINEMOD BOP evaluation
    │   ├── run_full_pipeline_ycbv.py         # YCB-Video BOP evaluation
    │   ├── run_full_pipeline_ho3d.py         # HO3D BOP evaluation
    │   ├── test_pipeline_functions.py        # 42 unit tests (no GPU needed)
    │   └── calibrated_system_lm.txt          # LLM system prompt backup (LM)
    │
    ├── visualization/           # Qualitative figure generation
    ├── experiments/             # Ablation studies
    ├── pipeline_test/           # Early calibration scripts
    ├── yoloe/                   # YOLOE standalone tests
    ├── foundationpose/          # FoundationPose dependency  ← DO NOT EDIT
    ├── sam2/                    # SAM2 segmentation          ← DO NOT EDIT
    ├── instantmesh/             # InstantMesh 3D recon       ← DO NOT EDIT
    └── bop_toolkit/             # BOP evaluation toolkit     ← DO NOT EDIT
```

> **Files marked DO NOT EDIT** are upstream dependencies. All project logic lives in `core/` and `pipeline_scripts/`.

---

## 4. Environment Setup

### Prerequisites

- NVIDIA GPU with CUDA 12.1 support (tested on NVIDIA L4, sm_89)
- Docker with NVIDIA Container Toolkit
- GCP VM: `g2-standard-4` (recommended) — see `master_vm.sh`

### Step 1 — Create the GCP VM (optional)

```bash
bash master_vm.sh
```

This provisions a `g2-standard-4` instance with 1× NVIDIA L4, Ubuntu 22.04, CUDA 12.9 drivers, and 4 TB SSD.

### Step 2 — Build the Docker image

```bash
# Full setup from scratch (clone, download weights, build image)
bash setup_any6d.sh

# Or just build the image if the repo is already present
bash build_any6d.sh
```

The Docker image `any6d:cuda12.1` includes:
- Python 3.9 conda environment (`Any6D`)
- PyTorch 2.4 with CUDA 12.1
- FoundationPose, SAM2, InstantMesh, BOP toolkit
- All C++ extensions pre-compiled for sm_89

Build time: approximately 30–60 minutes on first run.

### Step 3 — Start the container

```bash
docker compose -f Any6D/docker-compose.yml up -d
# Container name: any6d_active
```

### Step 4 — Set up the host environment (YOLOE)

YOLOE runs on the **host** machine (outside Docker) to avoid conflicting dependencies:

```bash
bash setup_master_env.sh
# Creates Python venv at ~/master_env/
# Installs: ultralytics, opencv-python, matplotlib, jupyter
```

Download the YOLOE model weights:

```bash
# Place the model at:
/workspace/yoloe/yoloe-26l-seg.pt
```

### Step 5 — Start Ollama (LLM)

```bash
# Install Ollama on the host
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Mistral model
ollama pull mistral:latest

# Verify it is accessible from inside Docker
# The container accesses Ollama at http://172.18.0.1:11434
```

---

## 5. Dataset Setup

All datasets are mounted into the Docker container at `/workspace/dataset/`.

### LINEMOD (LM)

BOP format. Download from the [BOP benchmark](https://bop.felk.cvut.cz/datasets/):

```
/workspace/dataset/lm/
├── models/                  # .ply mesh files (obj_000001.ply … obj_000015.ply)
├── models_eval/
├── test/
│   └── 000001/ … 000015/
│       ├── rgb/             # 640×480 PNG
│       ├── depth/           # 640×480 PNG (millimetres)
│       ├── mask/            # Binary GT masks
│       └── scene_gt.json    # Ground-truth poses
└── test_targets_bop19.json  # BOP evaluation targets
```

### YCB-Video (YCBV)

```
/workspace/dataset/ycbv/
├── models/                  # 21 object meshes
├── test/
│   └── 000048/ … 000059/
│       ├── rgb/
│       ├── depth/
│       ├── mask_visib/
│       └── scene_gt.json
└── test_targets_bop19.json
```

### HO3D

```
/workspace/dataset/ho3d/
└── evaluation/
    ├── MPM10/ MPM11/ … MPM14/
    ├── AP10/ AP11/ … AP14/
    ├── SB11/ SB13/
    └── SM1/
        ├── rgb/
        ├── depth/
        └── meta/            # Camera intrinsics + GT poses per frame
```

Each HO3D sequence also requires an **anchor frame** at:

```
/workspace/anchor_results/dexycb_reference_view_<OBJ>/
├── color.png
├── depth.png
├── ob_in_cam.txt            # 4×4 GT anchor pose
└── ob_mask.png
```

---

## 6. Core Module Reference

All shared logic lives in `Any6D/core/`. Import from inside Docker as:

```python
from core.constants import YOLOE_CONF, LLM_MODEL, ADD_THRESH_RATIO
from core.llm import call_llm, parse_llm
from core.detection import get_detection_mask, compute_iou
from core.pose_utils import rotation_error_deg, translation_error_cm
from core.metrics_utils import compute_frame_metrics, nanmean
from core.io_utils import save_result
```

---

### `core/constants.py`

Global configuration. Change these to adapt the pipeline to a different environment.

| Constant | Value | Purpose |
|---|---|---|
| `MM_TO_M` | `0.001` | Converts depth from mm (dataset storage) to metres (pose computation) |
| `ADD_THRESH_RATIO` | `0.10` | ADD success = error < 10% of object diameter (BOP standard) |
| `ANY6D_ITERS` | `5` | Number of Any6D refinement iterations per frame |
| `YOLOE_CONF` | `0.1` | Minimum detection confidence for YOLOE (low to maximise recall on small objects) |
| `YOLOE_MODEL` | `/workspace/yoloe/yoloe-26l-seg.pt` | YOLOE weights path inside Docker |
| `OLLAMA_URL` | `http://172.18.0.1:11434/api/generate` | Ollama REST endpoint (host bridge IP from inside Docker) |
| `LLM_MODEL` | `mistral:latest` | LLM model name served by Ollama |

---

### `core/llm.py`

**LLM keyword extraction via Ollama.**

```python
def call_llm(instruction: str, system_prompt: str, model: str = LLM_MODEL) -> str:
    """
    Send a user instruction to the Ollama LLM with a calibrated system prompt.

    Args:
        instruction:   Natural language user request (e.g. "hand me the mustard bottle").
        system_prompt: Dataset-specific CALIBRATED_SYSTEM string.
        model:         Ollama model name.

    Returns:
        Raw LLM text response (may be verbose; use parse_llm() to clean it).

    Raises:
        requests.HTTPError: If Ollama server is unreachable or returns an error.
    """
```

```python
def parse_llm(raw: str) -> str:
    """
    Extract a clean 1–3 word keyword from a raw LLM response.

    Handles these LLM output formats:
      - Arrow prefix:     "→ rubber duck"           → "rubber duck"
      - Quoted keyword:   'The keyword is "duck"'   → "duck"
      - Label prefix:     "keyword: benchvise"       → "benchvise"
      - Plain word:       "driller"                  → "driller"
      - Multiline:        first valid line is used

    Lines with more than 3 words are skipped (too verbose to be a keyword).

    Args:
        raw: Unprocessed LLM response string.

    Returns:
        Cleaned keyword string (1–3 words), or empty string if nothing valid found.
    """
```

---

### `core/detection.py`

**YOLOE open-vocabulary segmentation with singleton model loading.**

The model is loaded once and reused across all frames. This avoids the ~10-second GPU weight loading overhead on every call.

```python
def detect_mask(
    img_rgb: np.ndarray,
    keyword: str,
    H: int,
    W: int,
    conf: float | None = YOLOE_CONF,
    conf_fallbacks: tuple[float, ...] = (),
    use_first_det: bool = False,
    bgr_input: bool = False,
) -> tuple[np.ndarray | None, float]:
    """
    Run YOLOE open-vocabulary segmentation on a single image.

    Args:
        img_rgb:        Input image in RGB format, shape (H, W, 3).
        keyword:        Text prompt for YOLOE (e.g. "yellow mustard bottle").
        H, W:           Target mask height and width (used for resize after prediction).
        conf:           YOLOE confidence threshold. Pass None to use YOLOE's built-in default.
        conf_fallbacks: Additional thresholds tried in order if primary conf finds nothing.
                        Used for HO3D where objects can be heavily occluded: (0.05, 0.03).
        use_first_det:  If True, use masks.data[0] (first detection by index).
                        If False, use the detection with highest confidence score.
                        YCBV and HO3D use True; LINEMOD uses False.
        bgr_input:      If True, convert RGB → BGR before passing to YOLOE.
                        Required for HO3D because its images are loaded in BGR order.

    Returns:
        mask:  Binary boolean mask of shape (H, W), or None if nothing detected.
        score: YOLOE confidence score of the chosen detection (0.0 if none).
    """
```

```python
def get_detection_mask(
    img_rgb: np.ndarray,
    keyword: str,
    gt_mask: np.ndarray | None,
    H: int,
    W: int,
    conf: float | None = YOLOE_CONF,
    conf_fallbacks: tuple[float, ...] = (),
    use_first_det: bool = False,
    bgr_input: bool = False,
) -> tuple[np.ndarray | None, bool, float, float]:
    """
    Wrapper around detect_mask that also computes IoU with a ground-truth mask.

    Returns:
        mask:          YOLOE predicted mask (or None).
        yoloe_det:     True if YOLOE found a detection.
        conf_score:    YOLOE confidence (0.0 if no detection).
        iou:           IoU with gt_mask (-1.0 if gt_mask is None or no detection).
    """
```

```python
def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray | None) -> float:
    """
    Compute Intersection-over-Union between two binary masks.

    Args:
        pred_mask: Predicted binary mask, shape (H, W).
        gt_mask:   Ground-truth binary mask, shape (H, W), or None.

    Returns:
        IoU in [0, 1], or -1.0 if gt_mask is None.
    """
```

---

### `core/pose_utils.py`

**Pose geometry helpers.**

```python
def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Geodesic rotation error between two rotation matrices.

    Uses the formula: error = arccos((trace(R_pred.T @ R_gt) - 1) / 2)

    Args:
        R_pred: Predicted 3×3 rotation matrix.
        R_gt:   Ground-truth 3×3 rotation matrix.

    Returns:
        Angular error in degrees, in [0°, 180°].
    """
```

```python
def translation_error_cm(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """
    Euclidean translation error converted from metres to centimetres.

    Args:
        t_pred: Predicted translation vector [x, y, z] in metres.
        t_gt:   Ground-truth translation vector [x, y, z] in metres.

    Returns:
        L2 distance in centimetres.
    """
```

---

### `core/metrics_utils.py`

**BOP standard metrics computation.**

```python
def compute_frame_metrics(
    pred_pose: np.ndarray,
    gt_pose: np.ndarray,
    pts: np.ndarray,
    diameter: float,
    is_symmetric: bool,
    K: np.ndarray,
    H: int,
    W: int,
) -> dict:
    """
    Compute all BOP pose metrics for a single frame.

    Args:
        pred_pose:    Predicted 4×4 SE(3) pose matrix (metres).
        gt_pose:      Ground-truth 4×4 SE(3) pose matrix (metres).
        pts:          Object point cloud (N×3, metres), used for ADD/ADD-S.
        diameter:     Object bounding sphere diameter (metres), used as ADD threshold.
        is_symmetric: If True, compute ADD-S (symmetric) instead of ADD.
        K:            3×3 camera intrinsic matrix.
        H, W:         Image dimensions for MSSD/MSPD visibility mask.

    Returns:
        Dictionary with keys:
            "ADD":     1.0 if error < 10% diameter, else 0.0
            "ADD-S":   1.0 if min-distance error < 10% diameter (symmetric), else 0.0
            "AR":      Mean of MSSD and MSPD binary scores (BOP AR metric)
            "MSSD":    Maximum Symmetry-aware Surface Distance (raw value)
            "MSPD":    Maximum Symmetry-aware Projection Distance (raw value)
            "R_error": Rotation error in degrees
            "T_error": Translation error in centimetres
    """
```

```python
def nanmean(values: list[float | None]) -> float:
    """
    Arithmetic mean that silently ignores NaN and None entries.

    Args:
        values: List of floats, possibly containing NaN or None.

    Returns:
        Mean of valid values, or NaN if no valid values exist.
    """
```

---

### `core/io_utils.py`

**Results export to JSON and Excel.**

```python
def save_result(
    out_dir: str,
    obj_name: str,
    summary: dict,
    frames: list[dict],
) -> None:
    """
    Save per-object evaluation results to both JSON and XLSX.

    Files created:
        <out_dir>/<obj_name>_result.json   — full per-frame data + summary
        <out_dir>/<obj_name>_result.xlsx   — tabular view with MEAN row at bottom

    Args:
        out_dir:  Output directory path.
        obj_name: Object name used in filenames.
        summary:  Dict of aggregated metrics (ADD_mean, Det_Rate, R_med, etc.).
        frames:   List of per-frame metric dicts.
    """
```

---

## 7. Pipeline Scripts

### `pipeline_scripts/run_full_pipeline_linemod.py`

Full BOP evaluation on the LINEMOD dataset (15 objects, up to 1214 frames each).

**Key constants defined at module level:**

```python
LM_NAMES: dict[int, str]          # {1: "ape", 2: "benchvise", ..., 15: "phone"}
LM_SYMMETRIC: set[int]            # {10, 11}  — eggbox and glue
LM_INSTRUCTIONS: dict[int, str]   # Natural language instruction per object
CALIBRATED_SYSTEM: str            # 1195-example LLM system prompt
```

**YOLOE call for LINEMOD** uses argmax confidence (not first detection), with no BGR conversion:
```python
mask, detected, conf, iou = get_detection_mask(
    rgb, keyword, gt_mask, H, W,
    conf=YOLOE_CONF,        # 0.1 threshold
    use_first_det=False,    # argmax confidence
    bgr_input=False,        # images already in RGB
)
```

**CLI arguments:**

```bash
--obj_ids 1 5 8        # Only evaluate objects 1, 5, 8 (default: all 15)
--max_frames 200       # Cap frames per object (default: all)
--stride 5             # Evaluate every Nth frame (default: 1)
--llm_model mistral    # Ollama model name
--out_dir ./results/lm_pipeline
```

---

### `pipeline_scripts/run_full_pipeline_ycbv.py`

Full BOP evaluation on YCB-Video (21 objects across test scenes 48–59).

**Key difference from LM**: YOLOE uses the first detection (not argmax) and the built-in default confidence:

```python
mask, detected, conf, iou = get_detection_mask(
    rgb, keyword, gt_mask, H, W,
    conf=None,           # use YOLOE built-in default (no threshold passed)
    use_first_det=True,  # first detection by index, not highest confidence
)
```

**CLI arguments:**

```bash
--obj_ids 5 6          # Evaluate only objects 5 and 6
--scene_id 52          # Evaluate only scene 52
--max_frames 3         # Quick test with 3 frames
```

---

### `pipeline_scripts/run_full_pipeline_ho3d.py`

Evaluation on HO3D (13 sequences, hand-object interactions).

**Key difference from LM/YCBV**: HO3D requires anchor-based pose correction and has dataset-specific YOLOE settings:

```python
mask, detected, conf, iou = get_detection_mask(
    color, keyword, gt_mask_raw, H, W,
    conf=0.1,
    conf_fallbacks=(0.05, 0.03),  # retry with lower thresholds if nothing found
    use_first_det=True,
    bgr_input=True,               # HO3D images loaded in BGR order
)
```

**Anchor correction** applied after pose estimation:
```python
pred_pose_a = est.register(K, anchor_rgb, anchor_depth, anchor_mask, iteration=5)
pred_pose_q = est.register(K, color, depth, mask, iteration=5)
corrected_pose = (pred_pose_q @ np.linalg.inv(pred_pose_a)) @ gt_pose_a
```

**Metrics computed**: ADD-S, AR, MSSD, MSPD, VSD, R_error, T_error (full BOP suite including visibility-based VSD).

---

### `pipeline_scripts/main_pipeline.py`

Single-image inference. Use this for testing on arbitrary images without a dataset.

**Minimal mode** (direct pose, no anchor correction):

```python
# Outputs: pose_4x4.txt, result.json, viz_pipeline.png
python3 main_pipeline.py \
    --image /path/to/rgb.png \
    --depth /path/to/depth.png \
    --mesh  /path/to/object.ply \
    --K "572.4 0 325.2 0 572.4 242.0 0 0 1" \
    --instruction "Hand me the yellow mustard bottle"
```

**Anchor correction mode**:

```python
python3 main_pipeline.py \
    --image path/to/rgb.png \
    --depth path/to/depth.png \
    --mesh  path/to/object.ply \
    --K "572.4 0 325.2 0 572.4 242.0 0 0 1" \
    --instruction "Hand me the yellow mustard bottle" \
    --anchor_image path/to/anchor_rgb.png \
    --anchor_depth path/to/anchor_depth.png \
    --anchor_mask  path/to/anchor_mask.png \
    --anchor_pose  path/to/anchor_pose.txt    # 4×4 matrix, space-separated
```

**Output files:**

| File | Description |
|---|---|
| `pose_4x4.txt` | 4×4 SE(3) matrix, space-separated, metres |
| `result.json` | Instruction, keyword, YOLOE conf, pose, detection flag |
| `viz_pipeline.png` | 4-panel figure: RGB · Depth · YOLOE mask · Pose overlay |

---

## 8. Running Evaluations

All commands are executed **inside Docker**:

```bash
docker exec -it any6d_active bash
cd /workspace
```

### LINEMOD — full evaluation (all 15 objects)

```bash
/opt/conda/envs/Any6D/bin/python3 pipeline_scripts/run_full_pipeline_linemod.py
```

### LINEMOD — quick test (3 objects, 10 frames each)

```bash
/opt/conda/envs/Any6D/bin/python3 pipeline_scripts/run_full_pipeline_linemod.py \
    --obj_ids 5 8 12 --max_frames 10
```

### YCB-Video — full evaluation

```bash
/opt/conda/envs/Any6D/bin/python3 pipeline_scripts/run_full_pipeline_ycbv.py
```

### YCB-Video — single object, single scene, 3 frames

```bash
/opt/conda/envs/Any6D/bin/python3 pipeline_scripts/run_full_pipeline_ycbv.py \
    --obj_ids 5 --scene_id 52 --max_frames 3
```

### HO3D — full evaluation (all 13 sequences)

```bash
/opt/conda/envs/Any6D/bin/python3 pipeline_scripts/run_full_pipeline_ho3d.py
```

### Run in background, monitor progress

```bash
docker exec -d any6d_active bash -c \
    "/opt/conda/envs/Any6D/bin/python3 pipeline_scripts/run_full_pipeline_linemod.py \
     2>&1 | tee /tmp/lm_run.log"

# Check progress
docker exec any6d_active tail -20 /tmp/lm_run.log
```

> **Memory warning**: Do not run more than one dataset pipeline simultaneously. Running LM + YCBV + HO3D in parallel exhausts the 22 GB GPU VRAM and triggers OOM kills.

---

## 9. Single-Image Inference

Use `main_pipeline.py` to run the complete pipeline on **your own image** — no dataset evaluation loop, just one image in, one pose out.

### Inputs

| Argument | Required | Description |
|---|---|---|
| `--image` | ✅ | RGB image — `.png` or `.jpg` |
| `--depth` | ✅ | Depth map — `.png` uint16 (mm by default) |
| `--mesh` | ✅ | 3D object mesh — `.ply` |
| `--K` | ✅ | Camera intrinsics — 4 values `fx fy cx cy` or 9 values (3×3 row-major) |
| `--instruction` | ✅* | Natural language instruction, e.g. `"Hand me the mustard bottle"` |
| `--dataset` + `--obj_id` | ✅* | Alternative to `--instruction`: use a calibrated instruction from `lm`, `ycbv`, or `ho3d` |
| `--anchor_image` | ➕ | Anchor RGB — enables relative pose correction |
| `--anchor_depth` | ➕ | Anchor depth |
| `--anchor_mask` | ➕ | Anchor binary mask (optional — falls back to full image) |
| `--anchor_pose` | ➕ | Anchor ground-truth pose `.txt` (4×4 matrix) |
| `--depth_scale` | ➕ | Depth → metres scale (default `0.001`, i.e. mm → m) |
| `--skip_llm` | ➕ | Use `--instruction` directly as YOLOE keyword, skip Mistral |
| `--out_dir` | ➕ | Output directory (default `/tmp/main_pipeline_out`) |

*Either `--instruction` or `--dataset + --obj_id` is required.

---

### Case 1 — Minimal: your own image + free text prompt

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">docker exec any6d_active \
  /opt/conda/envs/Any6D/bin/python3 \
  /workspace/pipeline_scripts/main_pipeline.py \
  --image       /workspace/my_data/rgb.png \
  --depth       /workspace/my_data/depth.png \
  --mesh        /workspace/my_data/object.ply \
  --K           572.4 572.4 325.2 242.0 \
  --instruction "Hand me the yellow mustard bottle"</pre>

The `--K` field accepts **4 values** `fx fy cx cy` or **9 values** (full 3×3 row by row).

---

### Case 2 — With anchor correction (more accurate)

Provide a reference image of the same object with a known ground-truth pose. The pipeline corrects the predicted pose relative to this anchor.

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">docker exec any6d_active \
  /opt/conda/envs/Any6D/bin/python3 \
  /workspace/pipeline_scripts/main_pipeline.py \
  --image        /workspace/my_data/rgb.png \
  --depth        /workspace/my_data/depth.png \
  --mesh         /workspace/my_data/object.ply \
  --K            572.4 572.4 325.2 242.0 \
  --instruction  "Hand me the yellow mustard bottle" \
  --anchor_image /workspace/my_data/anchor_rgb.png \
  --anchor_depth /workspace/my_data/anchor_depth.png \
  --anchor_mask  /workspace/my_data/anchor_mask.png \
  --anchor_pose  /workspace/my_data/anchor_gt_pose.txt</pre>

The anchor pose file is a plain text 4×4 matrix (metres):
<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">0.9998  -0.0123   0.0156   0.1420
0.0121   0.9999   0.0089  -0.0531
-0.0157  -0.0087   0.9998   0.6230
0.0000   0.0000   0.0000   1.0000</pre>

---

### Case 3 — Use a calibrated dataset instruction

If your object is from LINEMOD, YCB-Video, or HO3D, use the pre-calibrated instruction and system prompt:

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">docker exec any6d_active \
  /opt/conda/envs/Any6D/bin/python3 \
  /workspace/pipeline_scripts/main_pipeline.py \
  --image   /workspace/my_data/rgb.png \
  --depth   /workspace/my_data/depth.png \
  --mesh    /dataset/lm/models/obj_000008.ply \
  --K       572.4 572.4 325.2 242.0 \
  --dataset lm \
  --obj_id  8</pre>

---

### Outputs

All files are written to `--out_dir` (default `/tmp/main_pipeline_out/`):

| File | Description |
|---|---|
| `pose_4x4.txt` | 4×4 SE(3) pose matrix in metres (plain text, loadable with `np.loadtxt`) |
| `result.json` | Full summary: instruction, keyword, YOLOE conf, anchor flag, pose |
| `result.xlsx` | Same as JSON but flat: R00…R22 columns + tx/ty/tz in centimetres |
| `viz_pipeline.png` | 4-panel figure: RGB \| Depth \| YOLOE mask overlay \| Pose axes (X=red, Y=green, Z=blue) |

**Console output example:**

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">Instruction : "Hand me the yellow mustard bottle"
[LLM] raw: "mustard bottle"
[LLM] → keyword: "mustard bottle"
Image: 640×480  |  Mesh: /workspace/my_data/object.ply
[YOLOE] Detected  (conf=0.74)
[Mesh] Vertices max > 10 → applied scale ×0.001 (mm→m)

── Pose 4×4 (metres) ──
  +0.998234  -0.012456  +0.015600  +0.142000
  +0.012100  +0.999900  +0.008900  -0.053100
  -0.015700  -0.008700  +0.999800  +0.623000
  +0.000000  +0.000000  +0.000000  +1.000000

  Translation: [+14.20cm, -5.31cm, +62.30cm]
  Rotation from identity: 2.1°

Pose saved → /tmp/main_pipeline_out/pose_4x4.txt
Saved → /tmp/main_pipeline_out/result.json  +  result.xlsx
Visualisation → /tmp/main_pipeline_out/viz_pipeline.png

Done. Results → /tmp/main_pipeline_out/</pre>

---

### Run the unit tests (no GPU needed)

`test_pipeline_functions.py` validates the full pipeline logic — LLM parsing, YOLOE wrappers, metrics, I/O helpers — without requiring a GPU, a dataset, or a running Docker container:

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">docker exec any6d_active \
  /opt/conda/envs/Any6D/bin/python3 \
  /workspace/pipeline_scripts/test_pipeline_functions.py -v</pre>

Or directly on the host if the Python path is set:

<pre style="background:#0d1117;color:#79c0ff;padding:1em;border-radius:8px;font-size:0.9em">cd Any6D/pipeline_scripts
python3 test_pipeline_functions.py -v</pre>

Expected result: **42 tests — 0 failures, 0 errors.**

Test classes:

| Class | What it covers |
|---|---|
| `TestLLMParsing` | `parse_llm()` — quoted text, arrow pattern, prefix stripping |
| `TestDetection` | `compute_iou()`, `detect_mask()` mock, fallback behaviour |
| `TestPoseUtils` | `rotation_error_deg()`, `translation_error_cm()` |
| `TestMetricsUtils` | `nanmean()`, `compute_frame_metrics()` on synthetic poses |
| `TestIOUtils` | `save_json()`, `build_summary_row()`, `save_result()` |
| `TestCalibratedSystem` | CALIBRATED_SYSTEM coverage: 1195+ examples, all LM/YCB/HOPE objects present |

---

### Using the core modules directly (Python API)

You can also call the pipeline steps individually from your own Python script, inside Docker:

```python
import sys
sys.path.insert(0, '/workspace')

from core.llm import call_llm, parse_llm
from core.detection import get_detection_mask
from core.constants import LLM_MODEL

import cv2
import numpy as np

# Step 1 — LLM keyword extraction
SYSTEM_PROMPT = "You are a visual keyword extractor for YOLOE..."
instruction = "Hand me the yellow mustard bottle on the table"
raw     = call_llm(instruction, SYSTEM_PROMPT, LLM_MODEL)
keyword = parse_llm(raw)
print(f"Keyword: {keyword}")   # → "mustard bottle"

# Step 2 — YOLOE segmentation
rgb = cv2.cvtColor(cv2.imread("rgb.png"), cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]
mask, detected, conf, iou = get_detection_mask(rgb, keyword, None, H, W)

if not detected:
    print("Object not detected — falling back to full-image mask")
    mask = np.ones((H, W), dtype=bool)

# Step 3 — Any6D pose estimation
from estimater import Any6D, ScorePredictor, PoseRefinePredictor
import trimesh, nvdiffrast.torch as dr

K     = np.array([[572.4, 0, 325.2], [0, 572.4, 242.0], [0, 0, 1.0]])
depth = cv2.imread("depth.png", cv2.IMREAD_ANYDEPTH).astype(np.float32) * 0.001
mesh  = trimesh.load("object.ply", force="mesh")

glctx = dr.RasterizeCudaContext()
est   = Any6D(mesh=mesh, scorer=ScorePredictor(),
              refiner=PoseRefinePredictor(), debug=0, glctx=glctx)

pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)

print("4×4 pose matrix (metres):")
print(pose)
np.savetxt("pose_4x4.txt", pose, fmt="%.8f")
```

---

## 10. Results & Metrics

### Metric definitions

| Metric | Definition | Unit |
|---|---|---|
| **ADD** | Mean point-to-point distance < 10% diameter | % frames |
| **ADD-S** | ADD with symmetric matching (min distance) | % frames |
| **AR** | Mean of MSSD and MSPD binary scores | % |
| **MSSD** | Max Symmetry-aware Surface Distance | cm |
| **MSPD** | Max Symmetry-aware Projection Distance | px |
| **R_med** | Median rotation error across frames | degrees |
| **T_med** | Median translation error across frames | cm |
| **Det_Rate** | % frames where YOLOE found a detection | % |
| **IoU** | Mean IoU between YOLOE mask and GT mask | [0, 1] |

### Example results

**HO3D sequences** (full pipeline LLM → YOLOE → Any6D + anchor correction):

| Sequence | Object | Det_Rate | ADD-S | AR | R_med |
|---|---|---|---|---|---|
| MPM10 | SPAM can | 91% | 92.5% | 45.0 | 3.2° |
| MPM13 | SPAM can | 95.5% | 93.6% | 38.0 | — |
| AP11 | Blue pitcher | **100%** | 57.5% | 28.3 | — |
| SB11 | Bleach bottle | **100%** | 72.5% | 50.4 | — |
| SM1 | Mustard bottle | 92.1% | **97.8%** | 32.9 | — |

**LINEMOD objects** (stride=5, best ADD with GT mask fallback):

| Object | Keyword (LLM) | Det_Rate | ADD |
|---|---|---|---|
| can | `"tin can"` | 0% | 100% (GT mask) |
| driller | `"yellow driller"` | 52% | 70% |
| holepuncher | `"hole puncher"` | 80% | 20% |

> LINEMOD detection rate is a known limitation — see [Section 13](#13-design-decisions--known-limitations).

### Output file format

After each object evaluation, two files are written to `results/<dataset>_pipeline/<timestamp>/`:

**`obj_000005_result.json`**:
```json
{
  "obj_id": 5,
  "obj_name": "can",
  "instruction": "Pick up that cylindrical steel can",
  "keyword": "tin can",
  "ADD_mean": 1.0,
  "Det_Rate": 0.0,
  "R_med": 1.2,
  "frames": [
    {
      "frame_id": 0,
      "yoloe_detected": false,
      "yoloe_conf": 0.0,
      "iou": -1.0,
      "ADD": 1.0,
      "R_error": 1.1,
      "T_error": 0.3
    }
  ]
}
```

**`obj_000005_result.xlsx`**: One row per frame + MEAN row at bottom, columns: frame_id, ADD, ADD-S, AR, R_error, T_error, yoloe_conf, iou.

---

## 11. Unit Tests

Tests cover all shared utilities and require **no GPU, no Docker, no dataset**.

```bash
# Run from project root (outside Docker)
# or inside Docker:
docker exec any6d_active \
    /opt/conda/envs/Any6D/bin/python3 -m pytest \
    /workspace/pipeline_scripts/test_pipeline_functions.py -v
```

**42 tests across 7 classes:**

| Test class | What is tested |
|---|---|
| `TestParseLlm` | LLM response parsing: arrow, quoted, prefix, multiline, empty |
| `TestRotationError` | Geodesic rotation error: identity, 90°, 180°, same matrix |
| `TestTranslationError` | Euclidean translation error: zero, 10 cm, diagonal |
| `TestNanmean` | NaN-aware mean: valid, mixed, all-NaN, empty, single |
| `TestComputeIoU` | IoU: perfect, zero, partial, None GT |
| `TestMetadata` | LM_NAMES, LM_INSTRUCTIONS, LM_SYMMETRIC, BOP constants |
| `TestCalibratedSystem` | System prompt: count ≥ 600, LM/YCB/HOPE/industrial coverage, txt match |

All 42 tests must pass before any pipeline change is merged.

---

## 12. LLM Keyword Calibration

### Why calibration is needed

YOLOE is an **open-vocabulary** segmentation model — it matches any text prompt against image regions. However, the matching quality depends heavily on the keyword:
- `"can"` → YOLOE finds 0% of LINEMOD cans (too generic)
- `"tin can"` or `"steel can"` → much better recall
- `"cylindrical steel can"` → highest confidence detections

The LLM is calibrated via a `CALIBRATED_SYSTEM` prompt that teaches it to produce YOLOE-friendly keywords.

### Structure of `CALIBRATED_SYSTEM`

Each dataset has its own independent `CALIBRATED_SYSTEM` string defined at the top of its pipeline script. They share the same format but are **not shared** between datasets — each is tuned to the visual characteristics of its objects.

Format:
```
You are a visual keyword extractor for YOLOE...
- 1 word is BEST when specific enough
- Use 2 words ONLY when ambiguous
- MAXIMUM 3 words
Examples:
"instruction phrase" → keyword
"instruction phrase" → keyword
...
```

### Coverage of `CALIBRATED_SYSTEM` (LINEMOD version)

The LINEMOD system prompt (`calibrated_system_lm.txt`) contains **1195 examples** covering:

| Source | Objects | Examples |
|---|---|---|
| LINEMOD (15 obj) | ape, benchvise, bowl, camera, can, cat, cup, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone | ~120 |
| YCB-Video (21 obj) | tomato soup can, mustard bottle, sugar box, tuna can, banana, bleach, mug, drill, scissors, marker, clamp, foam brick... | ~200 |
| HOPE grocery (28 obj) | alphabet soup, BBQ sauce, butter, cherries, chocolate pudding, granola, honey, macaroni, parmesan... | ~112 |
| HomebrewedDB (33 obj) | avocado, chips bag, ketchup, milk carton, monster energy can, pasta, pepsi, shampoo, yogurt... | ~132 |
| RU-APC Amazon (15 obj) | cheezit, clorox spray, expo marker, folgers coffee, oreo cookies, tissue box, tennis balls... | ~45 |
| T-LESS industrial (30 obj) | gray flat disc, gray cylinder, gray L bracket, gray gear, gray ring, gray knob... | ~90 |
| ITODD industrial (28 obj) | shiny metal bracket, aluminum L bracket, silver metal ring, metallic cylinder... | ~56 |
| ContactPose tools (25 obj) | flashlight, hammer, headphones, kitchen knife, frying pan, wine glass, screwdriver... | ~75 |
| HANDAL tools (17 obj) | adjustable wrench, pliers, spatula, ladle, wire whisk, rolling pin, tape dispenser... | ~51 |
| NOCS REAL275 (6 cat) | ceramic bowl, metal can, digital camera, laptop, water bottle, coffee mug | ~30 |
| Robotic / complex instructions | contextual, ambiguous, robotic command style, natural conversation | ~284 |

**Backup**: The full system prompt text is saved at `pipeline_scripts/calibrated_system_lm.txt` for version tracking. The canonical source is the `CALIBRATED_SYSTEM` variable in `run_full_pipeline_linemod.py`.

---

## 13. Design Decisions & Known Limitations

### YOLOE singleton pattern

The YOLOE model is loaded once at first call and cached as a module-level variable (`_yoloe_model`). Re-importing the module or calling from multiple threads would share the same instance. This is intentional: loading weights takes ~10 seconds and 4 GB of GPU memory. For multi-process evaluation, each process starts its own singleton.

### Dataset-specific YOLOE parameters

Each dataset requires different YOLOE call parameters because they were independently developed before the DRY refactor:

| Dataset | `conf` | `conf_fallbacks` | `use_first_det` | `bgr_input` |
|---|---|---|---|---|
| LINEMOD | 0.1 | — | False (argmax) | False |
| YCB-Video | None (default) | — | True (first) | False |
| HO3D | 0.1 | (0.05, 0.03) | True (first) | True |

Changing these would alter results. The values reproduce the original pipeline behaviour.

### LINEMOD detection is low for some objects

Objects like `can`, `ape`, `bowl` have near-zero YOLOE detection rates on LINEMOD. Root causes:
1. **Uniform background**: LINEMOD images have plain white backgrounds. YOLOE was trained on natural scene images and struggles with isolated objects.
2. **Single-word keywords**: The LLM often produces single-word outputs (`"ape"`, `"can"`) despite multi-word examples in the system prompt, because the instruction `"1 word is BEST"` dominates.
3. **Small objects**: Some LINEMOD objects (ape, cat) are very small relative to the image.

**Workaround**: When YOLOE fails to detect, the pipeline falls back to the GT mask for pose estimation. ADD scores remain valid in this mode but Det_Rate reflects the real detection capability.

### Anchor correction assumes rigid anchor

The anchor correction formula `corrected = (pred_q @ inv(pred_anchor)) @ gt_anchor` assumes the anchor pose estimation is accurate. If the anchor object is highly occluded or the anchor RGB-D quality is poor, the correction degrades.

### LLM temperature is not controlled

Ollama is called with default temperature. Responses for the same instruction may vary between runs. This introduces non-determinism in Det_Rate across runs but has negligible effect on pose metrics (which use the GT mask as fallback anyway).

---

## 14. Reproducing Results

### Requirements

1. Docker container `any6d_active` running with datasets mounted at `/workspace/dataset/`
2. Ollama with `mistral:latest` accessible at `http://172.18.0.1:11434`
3. YOLOE weights at `/workspace/yoloe/yoloe-26l-seg.pt`

### Can someone install this project without contacting the authors?

Yes — follow Sections 4 and 5. The main dependencies (Docker, Ollama, dataset download) are all publicly accessible.

### Can someone understand the functions without asking?

Yes — every function in `core/` has a docstring describing arguments, return values, and non-obvious behaviour (e.g. why `bgr_input=True` for HO3D).

### Can someone reproduce results?

Run the following to reproduce HO3D results:

```bash
docker exec any6d_active \
    /opt/conda/envs/Any6D/bin/python3 \
    /workspace/pipeline_scripts/run_full_pipeline_ho3d.py \
    2>&1 | tee /tmp/ho3d_repro.log
```

Run the following to reproduce LINEMOD results (3 representative objects):

```bash
docker exec any6d_active \
    /opt/conda/envs/Any6D/bin/python3 \
    /workspace/pipeline_scripts/run_full_pipeline_linemod.py \
    --obj_ids 5 8 12 \
    2>&1 | tee /tmp/lm_repro.log
```

Run the following to reproduce YCB-Video mustard bottle:

```bash
docker exec any6d_active \
    /opt/conda/envs/Any6D/bin/python3 \
    /workspace/pipeline_scripts/run_full_pipeline_ycbv.py \
    --obj_ids 6 \
    2>&1 | tee /tmp/ycbv_repro.log
```

Results appear in `./results/<dataset>_pipeline/<timestamp>/`. Compare JSON summaries with the values in Section 10.

---

## Glossary

| Term | Definition |
|---|---|
| **6D pose** | 3D rotation (SO(3)) + 3D translation (ℝ³) = rigid body transform SE(3) |
| **ADD** | Average Distance of Model Points — standard 6D pose success metric |
| **ADD-S** | ADD with symmetric model point matching (for rotationally symmetric objects) |
| **AR** | Average Recall — BOP benchmark aggregate score (mean of MSSD + MSPD) |
| **Anchor frame** | A single reference RGB-D image with known GT pose, used for relative pose correction |
| **BOP** | Benchmark for 6D Object Pose Estimation — standard evaluation protocol |
| **CALIBRATED_SYSTEM** | The few-shot LLM system prompt containing 1000+ instruction→keyword examples |
| **LM** | LINEMOD dataset (15 tabletop objects) |
| **MSSD** | Maximum Symmetry-aware Surface Distance |
| **MSPD** | Maximum Symmetry-aware Projection Distance |
| **Ollama** | Local LLM server — runs Mistral on the host machine |
| **Open vocabulary** | Detection/segmentation that works with any text description, not a fixed class list |
| **SE(3)** | Special Euclidean group in 3D — the space of rigid body transformations |
| **YCBV** | YCB-Video dataset (21 household objects, 12 test scenes) |
| **YOLOE** | Open-vocabulary variant of YOLO for instance segmentation |
