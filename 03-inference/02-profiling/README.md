# 2) Profiling Inference on MI300X (WeatherNet)

This lesson profiles the **inference** workload (multi-step rollout) to help you identify bottlenecks such as:
- compute-heavy convolution/GEMM kernels
- host↔device copies
- launch overhead / synchronization

We use:
- **PyTorch Profiler** (export Chrome trace)
- Optional: **`rocm-smi`** for real-time GPU utilization (run on the host; container availability varies)

## What is being profiled
- The workload is a **WeatherNet autoregressive rollout**: repeatedly calling the tiny CNN model `steps` times.
- The heavy compute is mainly from the convolution layers and the overall launch/synchronization pattern between steps.

## Checkpoint file (`.pt`)
- `profile.py` loads a `WeatherNet` checkpoint (by default from `../01-batch/weathernet_infer_demo.pt`).
- This `.pt` file is a **PyTorch state_dict** that contains the learned weights for the synthetic one-step map.
- If `--auto-train` is enabled and the checkpoint is missing, the script will train a small demo model first, then run the profiler.

## AMP in profiling
- The script supports `--use-amp` to enable autocast during inference.
- By default it uses `--amp-dtype bf16` (you can override it in `profile.py`).
- With AMP enabled, you may see reduced time in compute-heavy kernels, but you still want to look for other bottlenecks (copies, idle gaps, launch overhead).

---

## Files

- `profile.py` (updated) — profiles WeatherNet inference
- `submit_profile.sh` — SLURM launcher

---

## Run

```bash
cd 03-inference/02-profiling
sbatch submit_profile.sh
```

The profiler exports a trace file (Chrome tracing format) under `./traces/`.

---

## Optional: monitor the GPU live (ROCm)

In another terminal:

```bash
watch -n 1 rocm-smi
```

---

## Interpret results (quick guide)

| Symptom | Likely cause | What to try |
|---|---|---|
| Many small kernels / low GPU occupancy | Batch too small, poor fusion | Increase batch size, try AMP BF16 |
| High idle gaps in timeline | Synchronizations or data preparation overhead | Use `pin_memory` (if applicable) and reduce CPU work |
| Large “copy” time | Frequent host↔device transfers | Keep tensors on GPU during rollout |
| Compute dominates | Expect GEMM/conv kernels | Try `--use-amp` or `--compile-model` |

---

## Questions

1. In the trace, what are the top CUDA ops by self time?
2. Does enabling AMP reduce the main compute kernels’ time?
3. Do you see any CPU bottlenecks between forward passes?
