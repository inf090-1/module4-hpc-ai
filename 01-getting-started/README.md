# Day 1 — Getting Started

This day introduces the HPC cluster environment and establishes a single-GPU training baseline. You will inspect the GPU hardware, understand the differences between HPC and cloud-style AI infrastructure, learn about high-performance networking and storage, and run your first distributed training job.

## Learning Objectives

- Inspect GPU topology, VRAM, and compute capabilities on the INF0090 cluster
- Understand the differences between HPC (SLURM) and cloud (Kubernetes) approaches to AI training
- Recognize the role of InfiniBand and parallel filesystems (Lustre/BeeGFS) in HPC-AI
- Run a single-GPU PyTorch training job on AMD MI300X
- Launch a multi-GPU DDP training job with `torchrun` via SLURM

## Course Structure (Lessons)

| # | Lesson | Description |
|---|--------|-------------|
| 1 | [01-single-gpu-training](01-single-gpu-training/README.md) | Cluster inspection and MNIST single-GPU baseline |
| 2 | [02-distributed-intro](02-distributed-intro/README.md) | First distributed training job with DDP |

## Cluster Run Notes

All Python scripts require PyTorch with ROCm support. Install via pip inside a virtual environment:

```bash
module load python
python -m venv ~/venv-pytorch && source ~/venv-pytorch/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install lightning
```

**Important:** The first run will be slow (~5-10 min) due to HIP kernel JIT compilation. This is expected — subsequent runs will be fast.

When running on the GPU partition, use `srun` or `sbatch` with `--partition=gpu` and `--gpus-per-node=N`.

### INF090 Cluster Specs

| Component | Details |
|-----------|---------|
| GPU node | `g1` — 2× AMD MI300X (192 GB VRAM each) |
| CPU | AMD EPYC 9474F, 20 cores |
| RAM | 257 GB |
| ROCm | 7.2.0 |
| OS | Rocky Linux 9.7 |
