"""
Shared utilities for all Any6D dataset pipelines.

Module map
----------
constants    — single source of truth for thresholds, paths, and URLs
llm          — Ollama/Mistral call + response parser
detection    — YOLOE open-vocabulary segmentation (singleton model)
pose_utils   — anchor correction, rotation/translation error helpers
metrics_utils — BOP metrics: ADD, ADD-S, MSSD, MSPD, AR
io_utils     — JSON / XLSX result writers

Quick-start
-----------
A new dataset pipeline needs three things from this package:

1. An LLM system prompt (the CALIBRATED_SYSTEM string in each script).
   Call ``call_llm(instruction, CALIBRATED_SYSTEM)`` then ``parse_llm()``
   to get the YOLOE keyword.

2. ``get_detection_mask()`` to run YOLOE and fall back to GT when needed.

3. ``estimate_corrected_pose()`` for anchor-relative pose correction, then
   ``compute_frame_metrics()`` to evaluate against ground truth.

See ``pipeline_scripts/run_full_pipeline_linemod.py`` for a complete example.
"""
from .constants import MM_TO_M, OLLAMA_URL, LLM_MODEL, ADD_THRESH_RATIO, ANY6D_ITERS
from .llm import call_llm, parse_llm
from .detection import detect_mask, get_detection_mask, compute_iou
from .pose_utils import estimate_corrected_pose, rotation_error_deg, translation_error_cm
from .metrics_utils import nanmean, compute_frame_metrics
from .io_utils import save_json
