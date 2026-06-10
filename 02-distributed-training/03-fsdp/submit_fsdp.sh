#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:15:00
#SBATCH --output=fsdp-success-%j.out

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export TORCH_DISTRIBUTED_INIT_METHOD="env://"

echo "=== FSDP Demo: Expected Success (FSDP with massive model) ==="
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Strategy: fsdp (sharded across 2 GPUs)"
echo ""

srun --mpi=none apptainer exec --rocm \
  --bind "$WORKDIR:$WORKDIR" --pwd "$WORKDIR" \
  /opt/shared/rocm-pytorch.sif \
  python -u train_fsdp.py --devices 2 --strategy fsdp --d_model 4096 --num_layers 12 --seq_len 384 --batch_size 32 --ff_mult 16
