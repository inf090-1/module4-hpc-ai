#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=g1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=ddp-%j.out

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

export OMP_NUM_THREADS=4

# --- Rendezvous for PyTorch DDP (env://) ---
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
else
  export MASTER_ADDR=127.0.0.1
fi
export MASTER_PORT=29500
export TORCH_DISTRIBUTED_INIT_METHOD="env://"


srun apptainer exec --rocm \
  --bind "$WORKDIR:$WORKDIR" --pwd "$WORKDIR" \
  /opt/shared/rocm-pytorch.sif \
  python -u train_ddp.py --epochs 20 --seq_len 64 --batch_size 64 --num_workers 2
