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

# Optional non-container execution.
# Usage: `USE_VENV=1 sbatch submit_tuning.sh`
if [[ "${USE_VENV:-0}" == "1" ]]; then
  if [ -f "$HOME/venv-pytorch/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$HOME/venv-pytorch/bin/activate"
  fi
fi

echo "=== Performance Tuning (Single GPU) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""

run_cmd() {
  if [[ "${USE_VENV:-0}" == "1" ]]; then
    "$@"
  else
    apptainer exec --rocm \
      --bind "$WORKDIR:$WORKDIR" --pwd "$WORKDIR" \
      /opt/shared/rocm-pytorch.sif \
      "$@"
  fi
}

echo "--- Run 1: batch_size=8, num_workers=0 (Baseline) ---"
run_cmd python -u train_tuning.py --batch_size 8 --num_workers 0 --num_epochs 2
echo ""

echo "--- Run 2: batch_size=16, num_workers=4 (Optimized Loading) ---"
run_cmd python -u train_tuning.py --batch_size 16 --num_workers 4 --num_epochs 2
echo ""

echo "--- Run 3: Profiling with rocprof-sys (AMD) ---"
# We trace HIP and ROCTX. This requires rocprof-sys to be installed.
run_cmd rocprof-sys --roctx-trace --hip-trace -d profile_out python -u train_tuning.py --batch_size 16 --num_workers 4 --num_epochs 1
echo "Profiling output saved to profile_out directory."
