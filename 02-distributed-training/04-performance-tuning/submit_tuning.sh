#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=tuning-%j.out

module load python/3.13.1-gcc-11.5.0-linux-rocky9-ivybridge-33kdykh 2>/dev/null || true

export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:$LD_LIBRARY_PATH

source ~/venv-pytorch/bin/activate

echo "=== Performance Tuning (Single GPU) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""

echo "--- Run 1: batch_size=8, num_workers=0 (Baseline) ---"
srun --mpi=none bash -lc "source ~/venv-pytorch/bin/activate; python train_tuning.py --batch_size 8 --num_workers 0 --num_epochs 2"
echo ""

echo "--- Run 2: batch_size=16, num_workers=4 (Optimized Loading) ---"
srun --mpi=none bash -lc "source ~/venv-pytorch/bin/activate; python train_tuning.py --batch_size 16 --num_workers 4 --num_epochs 2"
echo ""

echo "--- Run 3: Profiling with rocprof-sys (AMD) ---"
# We trace HIP and ROCTX. This requires rocprof-sys to be installed.
srun --mpi=none bash -lc "source ~/venv-pytorch/bin/activate; rocprof-sys --roctx-trace --hip-trace -d profile_out python train_tuning.py --batch_size 16 --num_workers 4 --num_epochs 1"
echo "Profiling output saved to profile_out directory."
