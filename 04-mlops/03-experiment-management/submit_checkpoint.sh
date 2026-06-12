#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:15:00
#SBATCH --output=checkpoint-%j.out

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -1)
export MASTER_PORT=29500

echo "=== Distributed Checkpoint Training ==="
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Checkpoint dir: ./checkpoints"
echo ""

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINER_IMAGE=/opt/shared/rocm-pytorch.sif

srun --mpi=none apptainer exec --rocm \
     "$CONTAINER_IMAGE" \
     python train_checkpoint.py
