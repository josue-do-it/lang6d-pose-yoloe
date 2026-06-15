"""
Global constants shared by all pipeline scripts and the core/ module.

All numeric thresholds and external endpoints are defined here so that
a single change propagates to every dataset evaluation without hunting
through three separate pipeline files.

Do NOT edit dataset-specific behaviour here — use per-call overrides
(e.g. conf=None for YCBV, bgr_input=True for HO3D) in the caller instead.
"""

# Depth PNGs in LINEMOD / YCBV / HO3D are stored as uint16 millimetres.
# Any6D expects metres, so every depth load multiplies by this factor.
MM_TO_M = 0.001

# BOP standard success criterion: a pose is "correct" when the mean
# point-cloud distance between predicted and GT mesh is below 10% of the
# object's bounding-sphere diameter.  Raising this threshold inflates ADD
# scores; lowering it makes them stricter.  0.10 matches the BOP paper.
ADD_THRESH_RATIO = 0.10

# Number of Any6D pose-refinement iterations per frame.
# 5 is the value used in the Any6D paper (CVPR 2025).  Fewer iterations
# run faster but degrade accuracy on textured objects; more iterations
# give diminishing returns beyond ~7.
ANY6D_ITERS = 5

# Minimum YOLOE confidence to accept a segmentation mask.
# Kept intentionally low (0.1) because we prefer a false-positive mask
# over a missed detection: Any6D can recover a reasonable pose even with
# a noisy mask, but it cannot run at all without one.
# YCBV overrides this by passing conf=None (YOLOE's built-in default ≈ 0.25).
YOLOE_CONF = 0.1

# Absolute path to YOLOE weights inside the Docker container.
# On the host, the weights directory is mounted via docker-compose.yml.
YOLOE_MODEL = "/workspace/yoloe/yoloe-26l-seg.pt"

# Ollama REST endpoint reachable from inside the Docker container.
# 172.18.0.1 is the default Docker bridge gateway IP (the host machine).
# If Ollama is not listening on the bridge interface, update this or
# pass a custom URL to call_llm().
OLLAMA_URL = "http://172.18.0.1:11434/api/generate"

# Default LLM model served by Ollama.
# Mistral-7B gives ~1 s/call latency and extracts accurate 1-3 word keywords.
# Larger models (llama3, mixtral) improve quality marginally but add latency.
LLM_MODEL = "mistral:latest"
