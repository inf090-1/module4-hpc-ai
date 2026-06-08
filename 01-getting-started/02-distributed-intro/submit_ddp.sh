#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:15:00
#SBATCH --output=ddp-%j.out

module load python 2>/dev/null || true

export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:$LD_LIBRARY_PATH

source ~/venv-pytorch/bin/activate

echo "=== DDP MNIST Training ==="
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Job ID: $SLURM_JOB_ID"
echo ""

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -1)
export MASTER_ADDR
export MASTER_PORT=29500

srun --mpi=none bash -lc "export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:\$LD_LIBRARY_PATH; source ~/venv-pytorch/bin/activate; python train_ddp.py"
