#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=g1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=pp-%j.out

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

export OMP_NUM_THREADS=4

# Optional non-container execution.
# Usage: `USE_VENV=1 sbatch submit_pp.sh`
if [[ "${USE_VENV:-0}" == "1" ]]; then
  if [ -f "$HOME/venv-pytorch/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$HOME/venv-pytorch/bin/activate"
  fi
  srun --mpi=pmix python -u train_pp.py --epochs 20 --seq_len 64 --batch_size 64 --num_workers 2
else
  srun --mpi=pmix apptainer exec --rocm \
    --bind "$WORKDIR:$WORKDIR" --pwd "$WORKDIR" \
    /opt/shared/rocm-pytorch.sif \
    python -u train_pp.py --epochs 20 --seq_len 64 --batch_size 64 --num_workers 2
fi
