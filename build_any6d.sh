#!/bin/bash
# ============================================================
# build_any6d.sh
# Creates docker-compose.yml (if not exists) and builds Any6D
#
# Usage:
#   bash build_any6d.sh
# ============================================================
set -e

ANY6D_DIR="$HOME/open-vocabulary-6d-pose-yoloe/Any6D"

echo "========================================="
echo " Building Any6D Docker Image"
echo "========================================="

# ── 1. Go to Any6D directory ─────────────────────────────────
echo "[1/3] Navigating to Any6D directory..."
cd $ANY6D_DIR
echo "  → $(pwd) ✅"

# ── 2. Check / Create docker-compose.yml ─────────────────────
echo "[2/3] Checking docker-compose.yml..."
if [ -f "docker-compose.yml" ]; then
    echo "  → docker-compose.yml already exists, keeping it"
    # Just fix TORCH_CUDA_ARCH_LIST if wrong
    sed -i 's/TORCH_CUDA_ARCH_LIST=8.6/TORCH_CUDA_ARCH_LIST=8.9/' docker-compose.yml
    sed -i 's/TORCH_CUDA_ARCH_LIST=7.5/TORCH_CUDA_ARCH_LIST=8.9/' docker-compose.yml
    echo "  → TORCH_CUDA_ARCH_LIST verified for L4 (8.9) ✅"
else
    echo "  → docker-compose.yml not found, creating it..."
    cat > docker-compose.yml << 'COMPOSE'
services:
  any6d:
    build:
      context: .
      dockerfile: Dockerfile
    image: any6d:cuda12.1
    container_name: any6d

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    volumes:
      - .:/workspace

    stdin_open: true
    tty: true

    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
      - CONDA_DEFAULT_ENV=Any6D
      - TORCH_CUDA_ARCH_LIST=8.9

    shm_size: "16gb"
    working_dir: /workspace

    entrypoint: ["/bin/bash", "--login", "-c"]
    command: ["conda activate Any6D && exec bash"]
COMPOSE
    echo "  → docker-compose.yml created ✅"
fi

# ── 3. Build Docker image ─────────────────────────────────────
echo "[3/3] Building Docker image (this may take 30-60 min)..."
docker compose build

echo ""
echo "========================================="
echo " Build complete!"
echo ""
echo " To run the demo:"
echo "   cd $ANY6D_DIR"
echo "   docker compose run --rm any6d python run_demo.py"
echo ""
echo " To open a shell inside the container:"
echo "   docker compose run --rm any6d"
echo "========================================="
