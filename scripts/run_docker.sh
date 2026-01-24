#!/bin/bash

# Default dataset path from env or default
# Default dataset path from env or default
export DATASET_PATH=${DATASET_PATH:-/data/projects/nerf_data}
# Default GPU_ID to 'all' if not set
export GPU_ID=${GPU_ID:-all}

echo "Running CEM4DGS container..."
echo "  Dataset: $DATASET_PATH"
echo "  User: $(id -u):$(id -g) (${USER:-user})"
echo "  GPU(s):  $GPU_ID"

# Check if docker compose (plugin) is available
if docker compose version &> /dev/null; then
    CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    CMD="docker-compose"
else
    echo "Error: docker compose or docker-compose not found."
    exit 1
fi

# Auto-detect GPU architecture for build
if command -v nvidia-smi &> /dev/null; then
  # Get compute capability (e.g. 8.6, 8.9, 9.0)
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')
  if [[ ! -z "$ARCH" ]]; then
      # If Blackwell (12.0), append +PTX for safety
      if [[ "$ARCH" == "12.0" ]]; then
          ARCH="${ARCH}+PTX"
      fi
      # Determine arch list based on detected GPU to avoid building everything
      # We default to a safe list if detection fails or is weird, but here we override
      export TORCH_CUDA_ARCH_LIST="${ARCH}"
      echo "  Detected GPU Arch: $ARCH (Setting TORCH_CUDA_ARCH_LIST)"
  fi
fi

# Run the container
# We use 'env' to set UID/GID/d explicitly, avoiding bash's readonly UID variable issue
env UID=$(id -u) GID=$(id -g) USER=${USER:-user} GPU_ID=$GPU_ID TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} $CMD run --rm cem4dgs "$@"
