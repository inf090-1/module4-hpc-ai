#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=g1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:15:00
#SBATCH --output=ddp-%j.out

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

# --- Environment setup (matches typical NCCL/DDP SLURM patterns) ---
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load openmpi || true
fi

export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0

# --- Rendezvous for PyTorch DDP (env://) ---
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
else
  export MASTER_ADDR=127.0.0.1
fi
export MASTER_PORT=29500
export TORCH_DISTRIBUTED_INIT_METHOD="env://"


srun --mpi=pmix apptainer exec --rocm \
  --bind "$WORKDIR:$WORKDIR" --pwd "$WORKDIR" \
  /home/shared/rocm-pytorch.sif \
  python -u train_ddp.py --epochs 1 --max_batches 1 --seq_len 32 --batch_size 16 --num_workers 0
