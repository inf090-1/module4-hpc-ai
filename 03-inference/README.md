# Day 3 - Weather Forecasting Inference & Profiling (MI300X)

This day focuses on **ML inference** for weather forecasting on **AMD MI300X** GPUs, replacing the previous MNIST-focused demo.

You will run:
- **Autoregressive batch inference** with mixed precision (FP32 vs AMP/BF16)
- A **visualization example** that exports a **`.gif`** showing forecast evolution (2D heatmap + time-series)
- **GPU inference profiling** using PyTorch Profiler (exportable Chrome trace)

---

## Learning Objectives

- Run a multi-step weather forecast rollout with efficient batching
- Use PyTorch AMP (autocast) to accelerate inference on MI300X
- Export model predictions as a cartesian **GIF animation** for interpretability
- Profile inference to locate compute vs overhead bottlenecks

> Adapted in spirit from AMD's ROCm GeoArches / weather workflows for inference.
> https://rocm.blogs.amd.com/artificial-intelligence/geoarches-training/README.html

---

## Course Structure (Lessons)

| # | Lesson | Description |
|---|--------|-------------|
| 1 | [01-batch](01-batch/README.md) | WeatherNet inference + forecast GIF (`weather_forecast.gif`) with optional FP32/AMP benchmarking |
| 2 | [03-ensemble](03-ensemble/README.md) | Ensemble weather inference (mean + uncertainty) + GIF (`weather_forecast_ensemble.gif`) |
| 3 | [02-profiling](02-profiling/README.md) | PyTorch Profiler for inference workloads (Chrome trace export) |
| 4 | [01-apptainer-basics](../04-mlops-containers/01-apptainer-basics/README.md) | Apptainer containers for AI |
| 5 | [02-portability](../04-mlops-containers/02-portability/README.md) | AMD/NVIDIA portability |
| 6 | [03-experiment-management](../04-mlops-containers/03-experiment-management/README.md) | Distributed checkpoints |

---

## What’s in this `03-inference/` folder (quick summary)

- **`01-batch/`**: WeatherNet multi-step autoregressive inference → exports `weather_forecast.gif`.
- **`02-profiling/`**: Profiling inference (multi-step rollout) with PyTorch Profiler → exports traces.
- **`03-ensemble/`**: Ensemble inference (uncertainty) → exports `weather_forecast_ensemble.gif`.
- **`04-geoarches/`** *(optional advanced)*: Real ERA5 inference using **GeoArches/ArchesWeatherGen** inside Apptainer → exports `Z500_example.gif`.

---

## PyTorch Environment (ROCm)

Ensure your virtual environment has the required packages:

```bash
source ~/venv-pytorch/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```

For the GIF visualization you also need:

```bash
pip install matplotlib imageio
```

---

## Cluster Run Notes

All scripts target the GPU partition (typically **MI300X** on node `g1`). Use `sbatch` or `srun` with `--partition=gpu`.

---

## Optional advanced example: real ERA5 inference with GeoArches (Apptainer)

See:
- `03-inference/04-geoarches/README.md`

This is a heavier example (real dataset + pretrained model) aligned with AMD’s GeoArches/ArchesWeather blog, but using **Apptainer** instead of Docker.
