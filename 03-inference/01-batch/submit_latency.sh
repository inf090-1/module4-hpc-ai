#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=latency-sweep-%j.out

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:$LD_LIBRARY_PATH

echo "=== Latency/throughput sweep (WeatherNet) ==="
echo "GPU: $SLURM_GPUS_ON_NODE"
echo ""

srun --mpi=none apptainer exec --rocm \
  --bind "$WORKDIR:$WORKDIR" --pwd "$WORKDIR" \
  /opt/shared/rocm-pytorch.sif \
  python -u latency_sweep.py --use-amp --amp-dtype bf16 --forecast-steps 8
