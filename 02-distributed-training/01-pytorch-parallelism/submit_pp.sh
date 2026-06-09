#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=g1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:15:00
#SBATCH --output=pp-%j.out

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

# --- Environment setup ---
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load openmpi || true
fi

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0

srun --mpi=pmix apptainer exec --rocm \
  --bind "$WORKDIR:$WORKDIR" --pwd "$WORKDIR" \
  /home/shared/rocm-pytorch.sif \
  python -u train_pp.py --epochs 1 --max_batches 1 --seq_len 32 --batch_size 16 --num_workers 0
