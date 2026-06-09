#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:15:00
#SBATCH --output=lightning-cuda-%j.out

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

# --- Environment (similar to Lesson 1) ---
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export TORCH_DISTRIBUTED_INIT_METHOD="env://"

echo "=== Lightning DDP LLM Training (CUDA) ==="
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo ""

# Optional non-container execution.
# Usage: `USE_VENV=1 sbatch submit_lightning_cuda.sh`
if [[ "${USE_VENV:-0}" == "1" ]]; then
  if [ -f "$HOME/venv-pytorch/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$HOME/venv-pytorch/bin/activate"
  fi
  srun --mpi=none python -u train_lightning.py --devices 2 --strategy ddp
else
  srun --mpi=pmix apptainer exec --rocm \
    --bind "$WORKDIR:$WORKDIR" --pwd "$WORKDIR" \
    /opt/shared/rocm-pytorch.sif \
    python -u train_lightning.py --devices 2 --strategy ddp
fi
