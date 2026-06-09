#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=infer-ensemble-%j.out

module load python/3.13.1-gcc-11.5.0-linux-rocky9-ivybridge-33kdykh 2>/dev/null || true

export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:$LD_LIBRARY_PATH

source ~/venv-pytorch/bin/activate

echo "=== Ensemble Inference (WeatherNet) + GIF ==="
echo "GPU: $SLURM_GPUS_ON_NODE"
echo ""

srun --mpi=none bash -lc "export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:\$LD_LIBRARY_PATH; source ~/venv-pytorch/bin/activate; python ensemble_gif.py --use-amp --amp-dtype bf16 --ensemble 4 --forecast-steps 8 --var-idx 0 --out-gif weather_forecast_ensemble.gif"