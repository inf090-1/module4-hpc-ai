#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=infer-ensemble-%j.out

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:$LD_LIBRARY_PATH

echo "=== Sanity check: tostring_rgb usage in ensemble_gif.py ==="
grep -n "tostring_rgb" "ensemble_gif.py" || true

echo "=== Ensemble Inference (WeatherNet) + GIF ==="
echo "GPU: $SLURM_GPUS_ON_NODE"
echo ""

srun --mpi=none apptainer exec --rocm \
  --bind "$WORKDIR:$WORKDIR" --pwd "$WORKDIR" \
  /opt/shared/rocm-pytorch.sif \
  python -u ensemble_gif.py --use-amp --amp-dtype bf16 --ensemble 4 --forecast-steps 8 --var-idx 0 --out-gif weather_forecast_ensemble.gif
