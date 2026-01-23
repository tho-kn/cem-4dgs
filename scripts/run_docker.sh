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

# Run the container
# We use 'env' to set UID/GID/d explicitly, avoiding bash's readonly UID variable issue
env UID=$(id -u) GID=$(id -g) USER=${USER:-user} GPU_ID=$GPU_ID $CMD run --rm cem4dgs "$@"
