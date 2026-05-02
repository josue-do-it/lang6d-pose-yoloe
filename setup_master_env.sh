#!/bin/bash
# ============================================================
# setup_master_env.sh
# Creates master_env venv and installs YOLOE + Jupyter
#
# Usage:
#   bash setup_master_env.sh
# ============================================================
set -e

ENV_NAME="master_env"
PYTHON_VERSION="python3"

echo "========================================="
echo " Setting up $ENV_NAME"
echo "========================================="

# ── 1. Check Python ──────────────────────────────────────────
echo "[1/5] Checking Python..."
$PYTHON_VERSION --version || { echo "Python not found!"; exit 1; }

# ── 2. Create venv ───────────────────────────────────────────

echo "[2/5] Creating venv $ENV_NAME..."
if [ -f "$ENV_NAME/bin/activate" ]; then
    echo "  → $ENV_NAME already exists, reusing it"
else
    $PYTHON_VERSION -m venv $ENV_NAME
    echo "  → $ENV_NAME created ✅"
fi

# ── 3. Activate venv ─────────────────────────────────────────
echo "[3/5] Activating venv..."
source $ENV_NAME/bin/activate

# ── 4. Upgrade pip ───────────────────────────────────────────
echo "[4/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ── 5. Install dependencies ──────────────────────────────────
echo "[5/5] Installing packages..."

# PyTorch with CUDA 12.1
# pip install \
#     torch torchvision torchaudio \
#     --index-url https://download.pytorch.org/whl/cu121

# Jupyter + ipykernel
pip install jupyter ipykernel
pip install opencv-python matplotlib pillow
    # jupyterlab \
    # notebook \
    # ipykernel \
    # ipywidgets

# Register kernel in Jupyter
# python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"

# Core dependencies
pip install ultralytics
    # transformers \
    # supervision \
    # opencv-python \
    # Pillow \
    # numpy \
    # scipy \
    # matplotlib \
    # pandas \
    # tqdm \
    # requests \
    # huggingface-hub \
    # accelerate \
    # timm \
    # einops \
    

# # YOLOE Manual Install (try git first, then PyPI)
# pip install git+https://github.com/THU-MIG/yoloe.git || \
#     pip install yoloe || \
#     echo "WARNING: YOLOE not found - install manually if needed"

# echo ""
echo "========================================="
echo " Setup complete!"
# echo ""
# echo " To activate the venv:"
echo "   source $ENV_NAME/bin/activate"
echo ""
# echo " To launch Jupyter:"
# echo "   source $ENV_NAME/bin/activate"
# echo "   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
echo "========================================="
