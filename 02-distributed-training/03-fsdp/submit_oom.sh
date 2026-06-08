#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:10:00
#SBATCH --output=fsdp-oom-%j.out

module load python/3.13.1-gcc-11.5.0-linux-rocky9-ivybridge-33kdykh 2>/dev/null || true

export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:$LD_LIBRARY_PATH

source ~/venv-pytorch/bin/activate

echo "=== FSDP Demo: Expected OOM (DDP with massive model) ==="
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Strategy: ddp (single GPU — model too large)"
echo ""

srun --mpi=none bash -lc "export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:\$LD_LIBRARY_PATH; source ~/venv-pytorch/bin/activate; python train_fsdp.py --devices 1 --strategy ddp --d_model 4096 --num_layers 12"
