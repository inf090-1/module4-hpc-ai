# 1) Weather Forecast Inference (Batch + AMP) — with a GIF output

This lesson replaces the previous demo with a **weather forecasting** inference workflow aligned with the course theme.

You will:
- Run **autoregressive inference** with the (synthetic) **WeatherNet** model (rollout: `x[t+1] = model(x[t])`)
- Generate a **`.gif` animation** showing the forecast evolution as a **cartesian plot**:
  - 2D latitude/longitude heatmap (model output)
  - a time-series line plot for a single grid point
- Optionally compare **FP32 vs AMP** (BF16/FP16) throughput on **AMD MI300X**

> The underlying “ground truth” is the synthetic one-step operator used to create the dataset (so we can report an MSE over time).

## WeatherNet in this lesson
- **WeatherNet** is a tiny convolutional network that takes a gridded state `x` with shape `(B, C, nx, ny)` and outputs the next state with the same shape `(B, C, nx, ny)`.
- In other words, it learns a **one-step map** that approximates the synthetic physics operator.
- The rollout is then **autoregressive**: the model prediction at time `t` becomes the input at `t+1` (`x[t+1] = model(x[t])`).
- The script loads `--checkpoint` if it exists; if it does not, it will **auto-train a small demo** checkpoint on the synthetic one-step map so the inference lesson is runnable out of the box.

## What `infer_batch.py` actually does
- Creates a synthetic initial state batch `x0` from a deterministic formula (seed + sample index) on a grid of size `nx x ny`.
- Runs a rollout for `forecast_steps` time steps:
  - `x1 = model(x0)`
  - `x2 = model(x1)`
  - ... up to `x[forecast_steps]`
- Separately computes the **physics target** rollout using the synthetic one-step operator.
- Prints a **rollout MSE over time** comparing `states_pred[t]` vs `states_true[t]` for each time `t`.
- Renders a GIF:
  - heatmap = predicted field at each time step for `--var-idx`
  - line plot = value at a chosen grid point across time

## Checkpoint file (`.pt`)
- The file `weathernet_infer_demo.pt` (or whatever you pass via `--checkpoint`) is a **PyTorch checkpoint** containing the model **parameters** for `WeatherNet`.
- Concretely, it is a **`state_dict`** saved with `torch.save(model.state_dict(), ...)`.
- It is used to **skip the demo auto-training** step on subsequent runs, so inference and GIF generation start immediately.

## AMP optimization (`--use-amp`)
- `--use-amp` enables PyTorch **autocast** during inference so selected ops run in lower precision.
- `--amp-dtype bf16` uses **BF16** (bfloat16). BF16 is typically the preferred “fast + safe” mode on MI300X.
- What to expect: lower memory bandwidth / faster kernels (higher throughput), plus small numerical differences (so the rollout MSE may change slightly).
- In this script, AMP affects the rollout used for the GIF (and the optional benchmark).

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

## How this compares to the other 03-inference lessons
- `03-ensemble/`: keeps the same model, but uses the batch dimension as the “ensemble” (mean + uncertainty).
- `02-profiling/`: focuses on the compute cost of the rollout (timeline + kernel breakdown), not on visualization.
- `01-batch/latency_sweep.py`: focuses on how changing batch size changes throughput/latency.

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
- `[infer] Auto-training a small demo model ...` (only if the checkpoint is missing; subsequent runs reuse it)
- `[infer] Rollout MSE over time:`
- `[viz] Wrote GIF: weather_forecast.gif`

---

## Questions

1. How does AMP (BF16) change rollout throughput?
2. Does the forecast MSE monotonically increase with lead time?
3. Pick a different `--var-idx` and point (`--point-lat-idx/--point-lon-idx`): does the model improve/degrade in different regions?

## Notes on the plots
- The heatmap shows the **predicted** field at each forecast step for `--var-idx`.
- The line plot shows the value at a single grid point. If you do not pass `--point-lat-idx/--point-lon-idx`, the script uses the grid center: `nx//2, ny//2`.
