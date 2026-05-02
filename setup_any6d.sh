#!/bin/bash
# ============================================================
# setup_any6d.sh
# Clones Any6D, creates checkpoint folders, downloads weights,
# and builds the Docker image
#
# Usage:
#   bash setup_any6d.sh
# ============================================================
set -e

ANY6D_DIR="$HOME/open-vocabulary-6d-pose-yoloe/Any6D"

echo "========================================="
echo " Setting up Any6D"
echo "========================================="

# ── 1. Clone Any6D ───────────────────────────────────────────
echo "[1/6] Cloning Any6D..."
if [ -d "$ANY6D_DIR" ]; then
    echo "  → Any6D already exists, pulling latest..."
    cd $ANY6D_DIR && git pull
else
    git clone https://github.com/taeyeopl/Any6D.git $ANY6D_DIR
    echo "  → Any6D cloned ✅"
fi
cd $ANY6D_DIR

# ── 2. Create checkpoint directories ─────────────────────────
echo "[2/6] Creating checkpoint directories..."
mkdir -p foundationpose/weights/2024-01-11-20-02-45
mkdir -p foundationpose/weights/2023-10-28-18-33-37
mkdir -p sam2/checkpoints
mkdir -p instantmesh/ckpts
echo "  → Directories created ✅"

# ── 3. Download SAM2 checkpoint ──────────────────────────────
echo "[3/6] Downloading SAM2 checkpoint..."
SAM2_CKPT="sam2/checkpoints/sam2.1_hiera_large.pt"
if [ -f "$SAM2_CKPT" ]; then
    echo "  → SAM2 checkpoint already exists, skipping"
else
    wget -q --show-progress \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
        -O $SAM2_CKPT
    echo "  → SAM2 checkpoint downloaded ✅"
fi

# ── 4. Download InstantMesh checkpoints ──────────────────────
echo "[4/6] Downloading InstantMesh checkpoints..."
pip install -q huggingface-hub

INSTANT_DIR="instantmesh/ckpts"

if [ -f "$INSTANT_DIR/instant_mesh_large.ckpt" ]; then
    echo "  → InstantMesh checkpoints already exist, skipping"
else
    python3 -c "
from huggingface_hub import hf_hub_download
import os

ckpt_dir = '$INSTANT_DIR'
os.makedirs(ckpt_dir, exist_ok=True)

files = ['instant_mesh_large.ckpt', 'diffusion_pytorch_model.bin']
for f in files:
    print(f'  Downloading {f}...')
    hf_hub_download(
        repo_id='TencentARC/InstantMesh',
        filename=f,
        local_dir=ckpt_dir
    )
    print(f'  {f} downloaded')
"
    echo "  → InstantMesh checkpoints downloaded ✅"
fi

# ── 5. Download FoundationPose weights ───────────────────────
echo "[5/6] Downloading FoundationPose weights..."
pip install -q gdown

FOUND_DIR="foundationpose/weights"

if [ "$(ls -A $FOUND_DIR/2024-01-11-20-02-45)" ]; then
    echo "  → FoundationPose weights already exist, skipping"
else
    python3 -c "
import gdown
import os

# Download the weights folder from Google Drive
gdown.download_folder(
    'https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i',
    output='$FOUND_DIR',
    quiet=False
)
print('FoundationPose weights downloaded')
"
    echo "  → FoundationPose weights downloaded ✅"
fi
echo "  → Done✅"

# # ── 6. Build Docker image ─────────────────────────────────────
# echo "[6/6] Building Docker image..."

# # Fix TORCH_CUDA_ARCH_LIST for NVIDIA L4 (sm_89)
# sed -i 's/TORCH_CUDA_ARCH_LIST="8.6"/TORCH_CUDA_ARCH_LIST="8.9"/' Dockerfile
# sed -i 's/TORCH_CUDA_ARCH_LIST="7.5"/TORCH_CUDA_ARCH_LIST="8.9"/' Dockerfile
# echo "  → TORCH_CUDA_ARCH_LIST set to 8.9 for NVIDIA L4 ✅"

# # Install docker-compose if not present
# if ! command -v docker &> /dev/null; then
#     echo "Docker not found! Please install Docker first."
#     exit 1
# fi

# docker compose build
# echo "  → Docker image built ✅"

# echo ""
# echo "========================================="
# echo " Any6D setup complete!"
# echo ""
# echo " To run the demo:"
# echo "   cd $ANY6D_DIR"
# echo "   docker compose run --rm any6d python run_demo.py"
# echo "========================================="
