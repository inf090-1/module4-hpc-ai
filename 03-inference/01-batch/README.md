# 1) Weather Forecast Inference (Batch + AMP) — with a GIF output

This lesson replaces the old MNIST example with a **weather forecasting** inference workflow aligned with the course theme.

You will:
- Run **autoregressive inference** with the (synthetic) **WeatherNet** model
- Generate a **`.gif` animation** showing the forecast evolution as a **cartesian plot**:
  - 2D latitude/longitude heatmap (model output)
  - a time-series line plot for a single grid point
- Optionally compare **FP32 vs AMP** (BF16/FP16) throughput on **AMD MI300X**

> The underlying “ground truth” is the synthetic one-step operator used to create the dataset (so we can report an MSE over time).

---

## Files

- `infer_batch.py`
  - Runs multi-step rollout inference
  - Writes `weather_forecast.gif`
  - (Optional) benchmarks FP32 vs AMP rollouts
- `latency_sweep.py` (optional)
  - Sweeps **batch size** and measures **latency vs throughput** for rollout inference
  - Writes `latency_batch_sweep.csv` + `latency_batch_sweep.png`
- `submit_batch.sh` SLURM launcher
- `submit_latency.sh` SLURM launcher

---

## Install visualization dependencies (once)

```bash
pip install matplotlib imageio
```

---

## Quick start (local / interactive)

```bash
cd 03-inference/01-batch
python infer_batch.py --forecast-steps 8 --var-idx 0 --out-gif weather_forecast.gif
```

---

## Use BF16 AMP on MI300X

```bash
python infer_batch.py --use-amp --amp-dtype bf16
```

---

## Optional: `torch.compile` for inference optimization

```bash
python infer_batch.py --compile-model --use-amp
```

---

## Optional: latency vs batch-size sweep (direct)

```bash
python latency_sweep.py --use-amp --amp-dtype bf16 --forecast-steps 8 \
  --batch-sizes 1,2,4,8,16,32
```

> If `torch.compile` isn’t supported on your setup, the script will fall back to eager mode.

---

## Benchmark FP32 vs AMP rollout throughput

```bash
python infer_batch.py --benchmark --benchmark-steps 16 --use-amp
```

---

## SLURM run (recommended)

```bash
cd 03-inference/01-batch
sbatch submit_batch.sh
```

The output GIF will be created in the working directory.

---

## Optional: latency vs batch-size sweep

```bash
cd 03-inference/01-batch
sbatch submit_latency.sh
```

This writes:
- `latency_sweep_out/latency_batch_sweep.csv`
- `latency_sweep_out/latency_batch_sweep.png`

---

## Typical outputs

You should see something like:

- `[infer] Device: cuda:0`
- `[infer] Auto-training a small demo model ...` (only if the checkpoint is missing)
- `[infer] Rollout MSE over time:`
- `[viz] Wrote GIF: weather_forecast.gif`

---

## Questions

1. How does AMP (BF16) change rollout throughput?
2. Does the forecast MSE monotonically increase with lead time?
3. Pick a different `--var-idx` and point (`--point-lat-idx/--point-lon-idx`): does the model improve/degrade in different regions?