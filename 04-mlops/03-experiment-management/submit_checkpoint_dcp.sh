#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:15:00
#SBATCH --output=checkpoint-dcp-%j.out

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -1)
export MASTER_PORT=29500

echo "=== DCP Checkpoint Training ==="
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Checkpoint root: ./checkpoints_dcp"
echo ""

srun --mpi=none apptainer exec --rocm \
  /opt/shared/rocm-pytorch.sif \
  python train_checkpoint_dcp.py
