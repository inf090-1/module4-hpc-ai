# 2. AMD/NVIDIA GPU Portability

## The Problem

HPC clusters often have **mixed GPU hardware** — some nodes with AMD MI300X, others with NVIDIA A100/H100. As a researcher, you want your training code to run on **any GPU** without rewriting it.

This lesson teaches you how to write code and containers that work on both AMD and NVIDIA GPUs with minimal changes.

---

## Learning Objectives

By the end of this lesson you will be able to:
1. Explain the ROCm and CUDA software stacks and how they relate
2. Write device-agnostic PyTorch code that runs on any GPU
3. Build containers for both AMD and NVIDIA
4. Switch between GPU environments using Lmod modules on a cluster

---

## Software You Will Use

| Tool | What It Does | Where It Lives |
|------|-------------|---------------|
| **ROCm** | AMD's GPU compute platform (like CUDA for AMD) | Host system / inside container |
| **CUDA** | NVIDIA's GPU compute platform | Host system / inside container |
| **HIP** | AMD's CUDA-compatible API (translates CUDA → ROCm) | Inside ROCm |
| **RCCL** | AMD's collective communication library (drop-in NCCL replacement) | Inside PyTorch + ROCm |
| **NCCL** | NVIDIA's collective communication library for multi-GPU | Inside PyTorch + CUDA |
| **Lmod** | Environment module system for switching software versions | HPC cluster |

> **Reference**: [ROCm Documentation](https://rocm.docs.amd.com/) — AMD's official ROCm documentation.
> **Reference**: [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/) — NVIDIA's official CUDA documentation.

---

## How ROCm and CUDA Relate

Think of ROCm and CUDA as two different "languages" that talk to GPUs:

```
Your PyTorch code
       │
       ▼
   torch.cuda.* API          ← identical API for both!
       │
       ▼
┌──────────────┐    ┌──────────────┐
│   HIP layer  │    │   CUDA layer │
│  (AMD)       │    │  (NVIDIA)    │
├──────────────┤    ├──────────────┤
│   ROCm       │    │   CUDA       │
│   RCCL       │    │   NCCL       │
│   rocBLAS    │    │   cuBLAS     │
├──────────────┤    ├──────────────┤
│ AMD GPU      │    │ NVIDIA GPU   │
└──────────────┘    └──────────────┘
```

**Key insight**: AMD's **HIP** layer provides a CUDA-compatible API. This means `torch.cuda.is_available()` returns `True` on both AMD and NVIDIA, and `torch.cuda.get_device_name(0)` works on both.

---

## Software Stack Comparison

| Layer | AMD (ROCm) | NVIDIA (CUDA) | Notes |
|-------|-----------|---------------|-------|
| **Compute runtime** | ROCr (ROCm Runtime) | CUDA Driver | Talks to the GPU hardware |
| **Math libraries** | rocBLAS, rocFFT | cuBLAS, cuDNN | Matrix multiplication, convolutions |
| **Communication** | RCCL | NCCL | Multi-GPU collective ops (AllReduce, AllGather) |
| **Profiling** | rocprof, omniperf | nsight, nvprof, nsys | Performance analysis |
| **Container flag** | `--rocm` | `--nv` | Apptainer GPU passthrough |
| **PyTorch install** | `--index-url .../rocm7.2` | `--index-url .../cu121` | Different wheel indexes |

> **Reference**: [ROCm Architecture](https://rocm.docs.amd.com/en/latest/) — detailed AMD software stack.
> **Reference**: [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — NVIDIA's programming model.

---

## Step-by-Step: Writing Device-Agnostic Code

### The Goal

Write one training script that works on AMD, NVIDIA, or CPU without modification.

### Step 1 — Device detection

```python
import torch

def get_device():
    """Returns the best available device: CUDA/ROCm GPU, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")
```

**Why `torch.cuda` works for both**: When PyTorch is built with ROCm, it maps all `cuda` calls to `hip` calls under the hood. So `torch.cuda.is_available()` returns `True` on AMD GPUs, and `torch.cuda.get_device_name(0)` returns the AMD GPU name.

### Step 2 — Data movement

```python
# Create tensors and move them to GPU
x = torch.randn(1000, 1000).to(device)  # Works on both AMD and NVIDIA
y = torch.randn(1000, 1000).to(device)

# Matrix multiplication on GPU
z = torch.mm(x, y)
print(f"Result on {device}: {z.shape}")
```

### Step 3 — Multi-GPU setup

```python
import torch.distributed as dist

# This initializes communication for both NCCL (NVIDIA) and RCCL (AMD)
dist.init_process_group(backend="nccl")

# PyTorch uses "nccl" as the backend string for BOTH AMD and NVIDIA
# RCCL is designed to be API-compatible with NCCL
```

> **Reference**: [PyTorch Distributed Overview](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) — tutorial on distributed training.

---

## Step-by-Step: Switching Environments on the INF0090 Cluster

The cluster uses **Lmod** to manage software versions. You load modules to activate different environments.

### Step 1 — Check available modules

```bash
module avail 2>&1 | grep -i python
module avail 2>&1 | grep -i rocm
```

### Step 2 — Activate AMD ROCm environment

```bash
# Load Python
module load python/3.13.1-gcc-11.5.0-linux-rocky9-ivybridge-33kdykh

# Activate the ROCm virtual environment
source ~/venv-pytorch-rocm/bin/activate

# Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True AMD Instinct MI300X
```

### Step 3 — Activate NVIDIA CUDA environment (on a CUDA-equipped cluster)

```bash
# Load CUDA toolkit
module load cuda/12.1
module load python/3.13.1-gcc-11.5.0-linux-rocky9-ivybridge-33kdykh

# Activate the CUDA virtual environment
source ~/venv-pytorch-cuda/bin/activate

# Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True NVIDIA A100-SXM4-80GB
```

> **Reference**: [Lmod User Guide](https://lmod.readthedocs.io/) — how Lmod works, `module load`, `module avail`, `module spider`.

---

## Step-by-Step: Building Containers for Both GPUs

### Option A — Single container that detects GPU at runtime

```singularity
Bootstrap: docker
From: ubuntu:22.04

%post
    # Install both CUDA and ROCm (large image, but universal)
    apt-get update && apt-get install -y python3 python3-pip
    pip3 install torch torchvision

%runscript
    exec python3 "$@"
```

This approach is **simpler but produces a large image** (~10 GB with both stacks).

### Option B — Separate containers per GPU vendor (recommended)

```bash
# AMD container
cat > pytorch-rocm.def << 'EOF'
Bootstrap: docker
From: rocm/pytorch:latest

%post
    pip install lightning torchvision

%runscript
    exec python "$@"
EOF

# NVIDIA container
cat > pytorch-cuda.def << 'EOF'
Bootstrap: docker
From: pytorch/pytorch:latest

%post
    pip install lightning torchvision

%runscript
    exec python "$@"
EOF
```

Then run the appropriate one:

```bash
# On an AMD node
apptainer exec --rocm pytorch-rocm.sif python train.py

# On an NVIDIA node
apptainer exec --nv pytorch-cuda.sif python train.py
```

### Step-by-Step: Auto-detecting GPU type in a SLURM script

```bash
#!/bin/bash
#SBATCH --job-name=portable
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00

module load apptainer/1.4.1-gcc-11.5.0-linux-rocky9-ivybridge-olpavna

# Detect GPU vendor from SLURM
if nvidia-smi > /dev/null 2>&1; then
    GPU_FLAG="--nv"
    CONTAINER="pytorch-cuda.sif"
else
    GPU_FLAG="--rocm"
    CONTAINER="pytorch-rocm.sif"
fi

apptainer exec $GPU_FLAG \
  --bind $PWD:/workspace \
  --pwd /workspace \
  $CONTAINER \
  python train.py --devices 1
```

---

## NCCL vs RCCL: The Communication Layer

| Feature | NCCL (NVIDIA) | RCCL (AMD) |
|---------|---------------|------------|
| API | `ncclComm_t`, `ncclAllReduce` | **Same API** (compatible) |
| PyTorch backend string | `"nccl"` | `"nccl"` (yes, same name!) |
| Intranode interconnect | NVLink | Infinity Fabric |
| Internode interconnect | InfiniBand / RoCE | InfiniBand / RoCE |
| Library name | `libnccl.so` | `librccl.so` (but PyTorch links to it as `nccl`) |

**Why RCCL uses NCCL's API**: AMD designed RCCL as a **drop-in replacement** for NCCL so that PyTorch (and other frameworks) don't need code changes. The communication primitives (AllReduce, AllGather, ReduceScatter) have the same semantics; only the underlying hardware path differs.

> **Reference**: [RCCL Documentation](https://rocm.docs.amd.com/projects/rccl/en/latest/) — AMD's RCCL documentation.
> **Reference**: [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/) — NVIDIA's NCCL documentation.

---

## Practice Questions

1. **Why does RCCL use the same API and backend name as NCCL?** What would break if it used a different name?
2. **How would you write a SLURM script** that automatically detects the GPU type and uses the correct container flag?
3. **What are the trade-offs** of using a single universal container vs two separate containers?
4. **When would you choose ROCm over CUDA** (or vice versa) even if both are available?

---

## Further Reading

- [ROCm Getting Started](https://rocm.docs.amd.com/)
- [PyTorch ROCm Installation](https://pytorch.org/docs/stable/notes/rocm.html)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [AMD HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [CUDA to HIP Porting Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/user_guide/porting_guide.html)