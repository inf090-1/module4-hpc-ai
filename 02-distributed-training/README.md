# Day 2 — Distributed Training

This module covers distributed training at scale through four progressive lessons. We start with the three fundamental parallelism strategies in raw PyTorch (DDP, PP, TP), then show how PyTorch Lightning simplifies the code, introduce FSDP for models that exceed single-GPU memory, and finish with profiling and performance tuning.

---

## The Shared Architecture

All examples train a **GPT-like autoregressive Language Model** (decoder-only Transformer with causal masking) on the **Tiny Shakespeare** dataset.

- **Dataset**: A ~1MB text file of Shakespeare's works, tokenized at the **character level** (vocab size ~65). Each training sample is a sliding window of characters; the target is the same window shifted by one position (next-character prediction).
- **Model (Lesson 1: `TineLLM`)**: A small GPT-like decoder-only Transformer. It uses causal masking so each position can only use earlier characters. The defaults are tuned to keep the model small enough for the course to run on a couple of GPUs.
- **MassiveModel** (Lesson 3 only): A scaled-up version (`d_model=4096`, 12 layers, `dim_feedforward=16384`) designed to exceed single-GPU VRAM, demonstrating why FSDP is needed.

---

## Course Structure

| # | Lesson | Focus | Key Concept |
|---|--------|-------|-------------|
| 1 | [01-pytorch-parallelism](01-pytorch-parallelism/README.md) | Raw PyTorch DDP, PP, TP | How each parallelism strategy works under the hood |
| 2 | [02-lightning-ddp](02-lightning-ddp/README.md) | Lightning DDP | Eliminating distributed boilerplate |
| 3 | [03-fsdp](03-fsdp/README.md) | FSDP for large models | Sharding model state across GPUs |
| 4 | [04-performance-tuning](04-performance-tuning/README.md) | Profiling & tuning | Finding and fixing bottlenecks |

### How the Lessons Build on Each Other

```
Lesson 1: "How does distributed training actually work?"
  → DDP: replicate model, shard data, sync gradients
  → PP:  split model by layer, sequential GPU execution
  → TP:  split individual layers by tensor, all-gather for computation

Lesson 2: "How do I write less code for the same thing?"
  → Lightning automates all of DDP's boilerplate
  → Same training result, ~30 fewer lines of distributed plumbing

Lesson 3: "What if my model doesn't fit on one GPU?"
  → DDP replicates → OOM
  → FSDP shards → fits
  → One argument change in Lightning: strategy="ddp" → strategy="fsdp"

Lesson 4: "How do I make it faster?"
  → Batch size, num_workers, pin_memory
  → Profile with NVTX markers + nsys/rocprof-sys
  → Measure before optimizing
```

### Parallelism Strategies at a Glance

If you’re visual-thinking, here’s a common “3D parallelism” picture. It shows how the same training job can be split three ways:

- **Data Parallel (DDP)**: you have multiple copies of the model and split the *data* (different batches) across GPUs.
- **Tensor Parallel (TP)**: you split *one layer’s computation* across GPUs (each GPU holds only a slice of the layer tensors).
- **Pipeline Parallel (PP)**: you split the *model by layers* into stages and stream micro-batches through the stages.

![3D parallelism (data / tensor / pipeline)](https://www.deepspeed.ai/assets/images/3d-parallelism.png)

| | DDP | PP | TP | FSDP |
|--|-----|-----|-----|------|
| **What is split** | Data | Model (by layer) | Layers (by tensor) | Model state (params, grads, optimizer) |
| **Memory per GPU** | Full model | Model shard | Tensor shard | Sharded state |
| **Communication** | Gradient AllReduce | Activation P2P | AllReduce per sharded layer | AllGather + ReduceScatter per layer |
| **Bandwidth need** | Moderate | Low | High | Medium-High |
| **GPU utilization** | All active | Pipeline bubbles | All active (with overhead) | All active (with overhead) |
| **When to use** | Model fits 1 GPU | Layer-level split needed | Layer too large for 1 GPU | Model too large for 1 GPU |

---

## PyTorch Environment

```bash
source ~/venv-pytorch/bin/activate
pip install lightning
```

## Cluster Setup

All scripts target the GPU partition with 2× AMD MI300X on node `g1`. Use `sbatch` with `--partition=gpu`.

### Key SLURM Notes

1. **MASTER_ADDR/PORT**: All `submit_*.sh` scripts set these. Python scripts use `os.environ.setdefault()` so SLURM exports take precedence.
2. **srun for multi-GPU**: Use `srun --mpi=none bash -lc '...'` instead of bare `python` for distributed jobs. This lets SLURM manage process launching.
3. **Time limit**: Use at least `--time=00:10:00`. Shorter limits may not give enough time for model init + dataset loading on MI300X.

## Known Issues

- **TP stability**: Tensor Parallelism is sensitive to shapes and to how loss is computed across vocab shards. If you hit a runtime error, try lowering `--batch_size` and `--seq_len` first; then re-check the TP loss logic.
- **Node recovery**: If SLURM jobs fail hard, node `g1` may go to "down" state. Run `scontrol update nodename=g1 state=resume` (requires sudo) or wait for auto-recovery.
