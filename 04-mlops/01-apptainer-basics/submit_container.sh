#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=container-%j.out

# Auto-detect apptainer or singularity
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
else
    echo "ERROR: Neither apptainer nor singularity found"
    exit 1
fi

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONTAINER_IMAGE="${SCRIPT_DIR}/pytorch-rocm.sif"

if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "Container image not found: $CONTAINER_IMAGE"
    echo "Build it with: $CONTAINER_CMD build pytorch-rocm.sif pytorch-rocm.def"
    exit 1
fi

echo "=== Container Training ==="
echo "Container: $CONTAINER_CMD"
echo "Image: $CONTAINER_IMAGE"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo ""

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo $REPO_ROOT

$CONTAINER_CMD exec --rocm \
    --bind "${REPO_ROOT}:/workspace" \
    --pwd /workspace/02-distributed-training/02-lightning-ddp/ \
    "$CONTAINER_IMAGE" \
    python -u train_lightning.py \
        --devices 1 \
        --strategy auto \
        --max_epochs 2
