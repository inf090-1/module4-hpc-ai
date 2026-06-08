#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:15:00
#SBATCH --output=lightning-cuda-%j.out

module load cuda 2>/dev/null || true
module load python/3.13.1-gcc-11.5.0-linux-rocky9-ivybridge-33kdykh 2>/dev/null || true

export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:$LD_LIBRARY_PATH

source ~/venv-pytorch/bin/activate

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0

echo "=== Lightning DDP LLM Training (CUDA) ==="
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true
echo ""

srun --mpi=none bash -lc "export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:\$LD_LIBRARY_PATH; source ~/venv-pytorch/bin/activate; python train_lightning.py --devices 2 --strategy ddp"
