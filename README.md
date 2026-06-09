# Module 4 — HPC for AI: Distributed Training

This module covers training Artificial Intelligence models on HPC clusters. You will learn how to inspect GPU hardware, launch distributed training jobs across multiple GPUs, scale to models that exceed single-GPU memory, run inference at scale, and package your workflows with containers for reproducibility. The examples target AMD MI300X GPUs with ROCm, but CUDA/NVIDIA equivalents are noted throughout.

**What You Will Learn**

- How to inspect GPU topology on an HPC cluster
- How to train PyTorch models on a single GPU and scale to multi-GPU with DDP (Distributed Data Parallel)
- How to use PyTorch Lightning to simplify distributed training boilerplate
- How to use FSDP (Fully Sharded Data Parallel) for models that exceed single-GPU VRAM
- How to tune CPU/GPU affinity, batch size, and data loader workers for maximum throughput
- How to run batch inference with mixed precision for speedups
- How to profile training with PyTorch Profiler and ROCm tools to identify bottlenecks
- How to build and run Apptainer containers for portable AI workloads across AMD and NVIDIA clusters
- How to manage distributed checkpoints and experiment artifacts on parallel filesystems

**Lesson Overview**

- [01-getting-started](01-getting-started/README.md): cluster inspection, GPU topology, and a single-GPU MNIST training baseline.
- [02-distributed-training](02-distributed-training/README.md): multi-GPU parallelism strategies (DDP, Pipeline, Tensor), PyTorch Lightning DDP, FSDP for large models, and performance profiling with AMD/NVIDIA tools.
- [03-inference](03-inference/README.md): weather inference examples (batch + ensemble GIFs), inference profiling with PyTorch Profiler, and an optional real ERA5 inference workflow using GeoArches/ArchesWeatherGen inside Apptainer.
