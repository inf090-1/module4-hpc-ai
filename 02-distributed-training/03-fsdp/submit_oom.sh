#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:10:00
#SBATCH --output=fsdp-oom-%j.out

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0

echo "=== FSDP Demo: Expected OOM (DDP with massive model) ==="
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Strategy: ddp (single GPU — model too large)"
echo ""

srun --mpi=none apptainer exec --rocm \
  --bind "$WORKDIR:$WORKDIR" --pwd "$WORKDIR" \
  /opt/shared/rocm-pytorch.sif \
  python -u train_fsdp.py --devices 1 --strategy ddp --d_model 4096 --num_layers 12 --seq_len 384 --batch_size 32 --ff_mult 16
