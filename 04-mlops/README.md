# Day 4 — MLOps: Containers, Portability, Checkpoints & Experiment Tracking

## What is MLOps?

**MLOps** (Machine Learning Operations) is the practice of applying DevOps principles to machine learning workflows. It covers everything **after** you've written your first training script: making it reproducible (containers), portable (GPU portability), resilient (checkpointing), and trackable (experiment logging).

Think of it this way:
- **ML research**: "Can I make the model learn?"
- **MLOps**: "Can I make the model learn **reliably, reproducibly, and at scale**?"

> **Reference**: [MLOps: Continuous Delivery and Automation Pipelines in Machine Learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Google's overview of MLOps maturity levels.

### The MLOps Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MLOps Lifecycle                                   │
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  Data    │───▶│  Train   │───▶│  Deploy  │───▶│ Monitor  │──┐       │
│  │ Prepare  │    │  Model   │    │  Model   │    │ & Drift  │  │       │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │       │
│       ▲                                                        │       │
│       └────────────────────────────────────────────────────────┘       │
│                          (retrain on drift)                             │
│                                                                          │
│  This module covers:                                                    │
│  ├── 01 Containers    (reproducible environments)                       │
│  ├── 02 Portability   (run anywhere: AMD / NVIDIA)                      │
│  ├── 03 Checkpoints   (resilient training at scale)                     │
│  └── 04 MLflow        (track, compare, deploy)                          │
└─────────────────────────────────────────────────────────────────────────┘
```

> **Reference**: [MLOps Levels](https://ml-ops.org/content/mlops-principles) — overview of MLOps maturity from manual (Level 0) to fully automated (Level 2).

---

## Why This Module Exists

In the previous modules (distributed training, inference), you focused on **making the model work**. This module focuses on **making the workflow production-ready**:

| Problem | Lesson | Solution |
|---------|--------|----------|
| "It works on my machine but not the cluster" | 01-apptainer-basics | Bundle everything in a container |
| "It works on AMD but not NVIDIA" | 02-portability | Write device-agnostic code |
| "The job crashed and I lost 8 hours of training" | 03-experiment-management | Checkpoint to parallel filesystem |
| "I can't remember which run gave the best result" | 04-mlflow-tracking | Track experiments with MLflow |
| "How do I put it all together?" | 05-mlops-examples | End-to-end MLOps pipeline |

---

## Learning Objectives

By the end of this module you will be able to:
1. Build and run Apptainer containers for reproducible AI workloads on HPC
2. Write code that runs on both AMD (ROCm) and NVIDIA (CUDA) GPUs
3. Save and load distributed checkpoints correctly with DDP and FSDP
4. Track experiments, compare runs, and manage models with MLflow
5. Build an end-to-end MLOps pipeline with training, serving, and monitoring

---

## Course Structure

| # | Lesson | What You'll Learn | Key Software |
|---|--------|-------------------|-------------|
| 1 | [01-apptainer-basics](01-apptainer-basics/README.md) | Build and run containers without root | Apptainer |
| 2 | [02-portability](02-portability/README.md) | Write code that works on any GPU | ROCm, CUDA, HIP, RCCL |
| 3 | [03-experiment-management](03-experiment-management/README.md) | Save training progress at scale | PyTorch DCP, FSDP |
| 4 | [04-mlflow-tracking](04-mlflow-tracking/README.md) | Track experiments and version models | MLflow |
| 5 | [05-mlops-examples](05-mlops-examples/README.md) | End-to-end MLOps pipeline | MLflow, FastAPI, Docker |

### How the Lessons Build on Each Other

```
Lesson 1: "How do I make my environment reproducible?"
  → Apptainer bundles runtime + dependencies into a single .sif file
  → No root required; works natively with SLURM
  → Reference: Apptainer is used by NASA, CERN, and most national labs

Lesson 2: "How do I run the same container on AMD and NVIDIA?"
  → ROCm and CUDA have different runtimes but compatible PyTorch APIs
  → HIP provides CUDA compatibility layer for AMD
  → Use `--rocm` or `--nv` flag; write device-agnostic code

Lesson 3: "How do I save/resume training at scale?"
  → DDP: rank 0 saves full model; FSDP: all ranks save shards via DCP
  → Use parallel filesystem (/scratch) for high-bandwidth I/O
  → Save at regular intervals to recover from crashes

Lesson 4: "How do I track experiments and deploy models?"
  → MLflow logs params/metrics/artifacts per run
  → Model Registry versions models with lifecycle stages
  → Compare runs in a web UI to find the best configuration

Lesson 5: "How do I put it all together?"
  → End-to-end pipeline: train → serve → monitor
  → Docker Compose orchestrates MLflow + FastAPI services
  → Data drift detection with Evidently AI
  → Complete example you can extend for your own projects
```

---

## MLOps Workflow at a Glance

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   01 Container  │────▶│  02 Portability  │────▶│  03 Checkpoints  │
│   (Apptainer)   │     │  (ROCm/CUDA)     │     │  (DDP/FSDP)      │
└─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────┐
                                               │  04 MLflow       │
                                               │  (Tracking/Reg)  │
                                               └────────┬─────────┘
                                                        │
                                                        ▼
                                               ┌──────────────────┐
                                               │  05 End-to-End   │
                                               │  MLOps Pipeline  │
                                               └──────────────────┘
```

| Component | Tool | Purpose |
|-----------|------|---------|
| **Environment** | Apptainer | Reproducible, rootless containers |
| **Hardware** | ROCm / CUDA | Vendor-agnostic GPU execution |
| **State** | `torch.distributed.checkpoint` | Scalable DDP/FSDP checkpoints |
| **Tracking** | MLflow | Experiments, artifacts, model registry |
| **Pipeline** | FastAPI + Docker Compose | End-to-end training, serving, monitoring |

---

## Prerequisites

Before starting this module, you should:
- Be familiar with basic PyTorch (tensors, models, training loops)
- Have completed Day 2 (distributed training) — you'll use DDP/FSDP concepts
- Have access to the INF0090 HPC cluster with GPU nodes

---

## Environment Setup

### Step 1 — Activate the PyTorch environment

```bash
# For AMD MI300X GPUs
source ~/venv-pytorch-rocm/bin/activate

# Install MLflow (needed for Lesson 4)
pip install mlflow
```

### Step 2 — Load Apptainer

```bash
# Required for Lessons 1-2
module load apptainer/1.4.1-gcc-11.5.0-linux-rocky9-ivybridge-olpavna

# Verify
apptainer --version
```

### Step 3 — Set up Apptainer cache (first time only)

```bash
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer/cache
export APPTAINER_TMPDIR=/scratch/$USER/apptainer/tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
```

### Step 4 — Set MLflow tracking URI (Lesson 4)

```bash
# Run on login node: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
# Then on compute node:
export MLFLOW_TRACKING_URI=http://<login-node>:5000
```

---

## Cluster Setup

All scripts target the GPU partition with AMD MI300X on node `g1`.

### Key SLURM Notes

| Topic | Details |
|-------|---------|
| **Apptainer module** | Load before building/running containers |
| **GPU flags** | Use `--rocm` for AMD, `--nv` for NVIDIA in `apptainer exec` |
| **Parallel FS** | Save checkpoints to `/scratch` for bandwidth |
| **MLflow server** | Run on login node, set `MLFLOW_TRACKING_URI` before `sbatch` |
| **Time limit** | Use `--time=00:10:00` minimum; container builds take longer |

### Environment Variables

```bash
# MLflow tracking (set before sbatch)
export MLFLOW_TRACKING_URI=http://<login-node>:5000

# Apptainer cache (avoids filling /home)
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer/cache
export APPTAINER_TMPDIR=/scratch/$USER/apptainer/tmp
```

---

## Reference Table: All Software Used

| Software | Version/Source | Purpose | Lesson |
|----------|---------------|---------|--------|
| **Apptainer** | 1.4.1 (cluster module) | Rootless containers | 01 |
| **ROCm** | Latest (in container) | AMD GPU compute platform | 01, 02 |
| **CUDA** | 12.x (in container) | NVIDIA GPU compute platform | 02 |
| **HIP** | Part of ROCm | CUDA-compatible API for AMD | 02 |
| **RCCL** | Part of ROCm | Multi-GPU communication for AMD | 02 |
| **NCCL** | Part of CUDA | Multi-GPU communication for NVIDIA | 02 |
| **PyTorch** | Latest | Deep learning framework | All |
| **PyTorch Lightning** | Latest | Training boilerplate reduction | 01 |
| **torch.distributed.checkpoint** | PyTorch built-in | Distributed checkpoint format | 03 |
| **MLflow** | 2.12.2 | Experiment tracking & model registry | 04, 05 |
| **SQLite** | System | MLflow backend database (dev) | 04, 05 |
| **FastAPI** | 0.104.1 | REST API framework for model serving | 05 |
| **scikit-learn** | 1.5.2 | Machine learning (RandomForest) | 05 |
| **Evidently AI** | 0.3.2 | Data drift detection and reporting | 05 |
| **Docker Compose** | Latest | Multi-container orchestration | 05 |
| **SLURM** | Cluster scheduler | Job submission and scheduling | All |
| **Lustre/BeeGFS** | Cluster filesystem | High-bandwidth parallel storage | 03 |
| **Lmod** | Cluster module system | Software version management | 02 |

---

## Known Issues

| Issue | Workaround |
|-------|-----------|
| Container build time (first run) | Cache on `/scratch` with `APPTAINER_CACHEDIR` |
| MLflow artifact store is local-only | Use S3/MinIO for multi-node production |
| FSDP + MLflow logging | Gather full state dict on rank 0 before logging |
| Port 5000 in use | Change MLflow port and update `MLFLOW_TRACKING_URI` |
| SLURM node goes "down" | `scontrol update nodename=g1 state=resume` (sudo) |