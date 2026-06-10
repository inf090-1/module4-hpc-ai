#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=tuning-%j.out

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

export OMP_NUM_THREADS=4

echo "=== Performance Tuning (Single GPU) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""

APPTAINER="apptainer exec --rocm --bind $WORKDIR:$WORKDIR --pwd $WORKDIR /opt/shared/rocm-pytorch.sif"
run_cmd() {
  $APPTAINER "$@"
}

echo "--- Run 1: batch_size=8, num_workers=0 (Baseline) ---"
run_cmd python -u train_tuning.py --batch_size 8 --num_workers 0 --num_epochs 2
echo ""

echo "--- Run 2: batch_size=16, num_workers=4 (Optimized Loading) ---"
run_cmd python -u train_tuning.py --batch_size 16 --num_workers 4 --num_epochs 2
echo ""

echo "--- Run 3: Profiling with rocprofv3 (AMD) ---"
# rocprofv3 produces richer traces than rocprof-sys.
run_cmd rocprofv3 \
  --stats --kernel-trace --hip-runtime-trace --memory-copy-trace \
  --summary --summary-output-file rocprofv3_summary.txt \
  --output-directory rocprofv3_out \
  --output-file rocprofv3_out \
  -f pftrace \
  -- python -u train_tuning.py --batch_size 16 --num_workers 4 --num_epochs 1
echo "Profiling output saved to rocprofv3_out directory."
