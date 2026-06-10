# 1. Apptainer Containers for AI

## Why Containers?

When you work on a research project, you install specific versions of Python, PyTorch, and other libraries. Weeks later, a teammate runs your code and gets a different result — or a crash — because their environment is slightly different. This is the classic "works on my machine" problem.

A **container** bundles your entire runtime (OS libraries, Python, packages, even the training script) into a single, immutable file. Anyone with the same container runs the same code and gets the same result — regardless of what's installed on the host machine.

**Apptainer** (formerly Singularity) is the container runtime designed for HPC clusters. Unlike Docker, it:
- Runs **without root privileges** (critical on shared clusters)
- Integrates natively with **SLURM** job schedulers
- Mounts the host filesystem transparently (no slow overlay layers)
- Supports **GPU passthrough** for AMD (`--rocm`) and NVIDIA (`--nv`)

### Container Architecture: Docker vs Apptainer

```
┌─────────────────────────────────────────────────────────────────┐
│                        DOCKER                                    │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│   │ Container│    │ Container│    │ Container│   ← Each isolated │
│   │    A     │    │    B     │    │    C     │     with daemon   │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘                  │
│        │               │               │                         │
│   ┌────▼───────────────▼───────────────▼─────┐                  │
│   │          Docker Daemon (root)             │   ← Requires root│
│   └──────────────────┬───────────────────────┘                  │
│                      │                                           │
│   ┌──────────────────▼───────────────────────┐                  │
│   │              Host OS (Linux)              │                  │
│   └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      APPTAINER                                   │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│   │ Container│    │ Container│    │ Container│   ← Each runs as  │
│   │    A     │    │    B     │    │    C     │     your user     │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘                  │
│        │               │               │                         │
│   ┌────▼───────────────▼───────────────▼─────┐                  │
│   │     Apptainer (single binary, no daemon)  │  ← No root needed│
│   └──────────────────┬───────────────────────┘                  │
│                      │                                           │
│   ┌──────────────────▼───────────────────────┐                  │
│   │              Host OS (Linux)              │                  │
│   └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

> **Key difference**: Docker requires a root daemon running in the background. Apptainer is a single binary that runs containers as your user — no daemon, no root. This makes it safe for shared HPC clusters where multiple users share the same nodes.

> **Reference**: [Apptainer User Guide](https://docs.apptainer.org/en/latest/user_guide.html) — official documentation covering all commands and options.

---

## Learning Objectives

By the end of this lesson you will be able to:
1. Explain why containers matter for reproducible AI research
2. Write an Apptainer definition file (`.def`) for a PyTorch + ROCm environment
3. Build a container image (`.sif`)
4. Run training inside a container via SLURM

---

## Software You Will Use

| Tool | What It Does | Installed On |
|------|-------------|--------------|
| **Apptainer** | Build and run containers without root | HPC cluster (via `module load`) |
| **PyTorch** | Deep learning framework | Inside the container |
| **ROCm** | AMD GPU compute platform | Inside the container (base image) |
| **SLURM** | HPC job scheduler | HPC cluster (login/compute nodes) |

---

## Apptainer vs Docker on HPC

| Feature | Docker | Apptainer |
|---------|--------|-----------|
| Root required | Yes (daemon must run as root) | No (runs as your user) |
| GPU access | `--gpus all` | `--rocm` (AMD) / `--nv` (NVIDIA) |
| Filesystem | Overlay (slow on parallel FS) | Native bind mount (fast) |
| SLURM integration | Poor (daemon-based) | Excellent (single-binary execution) |
| Registry | Docker Hub | Docker Hub + Singularity/Apptainer |

> **Reference**: [Apptainer vs Docker](https://docs.apptainer.org/en/latest/admin_guide/admin_quickstart.html#docker-compatibility) — how Apptainer handles Docker images.

---

## Dockerfile vs Apptainer Definition File: Side-by-Side

This lesson includes both a `Dockerfile` and a `.def` file that do **the same thing**: build a PyTorch + ROCm container with `lightning` and `torchvision` installed.

### Dockerfile (Docker)

```dockerfile
FROM rocm/pytorch:latest

LABEL author="INFO090"
LABEL description="PyTorch with ROCm for HPC-AI course"

RUN pip install --no-cache-dir lightning torchvision

ENTRYPOINT ["python"]
```

### Apptainer Definition File (.def)

```singularity
Bootstrap: docker
From: rocm/pytorch:latest

%labels
    Author INFO090
    Description PyTorch with ROCm for HPC-AI course

%post
    pip install --no-cache-dir lightning torchvision

%runscript
    exec python "$@"
```

### Line-by-Line Comparison

| Concept | Dockerfile | Apptainer .def | Notes |
|---------|-----------|---------------|-------|
| **Base image** | `FROM rocm/pytorch:latest` | `Bootstrap: docker` + `From: rocm/pytorch:latest` | Apptainer can pull from Docker Hub directly |
| **Metadata** | `LABEL author="INFO090"` | `%labels` section | Same purpose, different syntax |
| **Install packages** | `RUN pip install ...` | `%post` section | Both run commands during build |
| **Default command** | `ENTRYPOINT ["python"]` | `%runscript` → `exec python "$@"` | Apptainer's `$@` passes CLI args through |
| **Build command** | `docker build -t image .` | `apptainer build image.sif def.def` | Docker produces layers; Apptainer produces a single `.sif` file |
| **Run command** | `docker run --gpus all image` | `apptainer exec --rocm image.sif` | No daemon needed for Apptainer |

### Key Differences

```
┌─────────────────────────────────────────────────────────────────┐
│                     DOCKER BUILD                                 │
│                                                                  │
│   Dockerfile ──▶ docker build ──▶ Layers ──▶ Registry/daemon    │
│                                                                  │
│   • Produces layered image (multiple files)                      │
│   • Requires Docker daemon running (root)                        │
│   • Push to Docker Hub for sharing                               │
│   • Run with: docker run --gpus all image python train.py        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   APPTAINER BUILD                                │
│                                                                  │
│   .def file ──▶ apptainer build ──▶ Single .sif file            │
│                                                                  │
│   • Produces single SquashFS file (immutable, portable)         │
│   • No daemon needed (runs as your user)                        │
│   • Copy the .sif file to share                                 │
│   • Run with: apptainer exec --rocm image.sif python train.py   │
└─────────────────────────────────────────────────────────────────┘
```

### When to Use Which?

| Scenario | Use | Why |
|----------|-----|-----|
| **Running on HPC cluster** | Apptainer `.def` | No root, no daemon, SLURM integration |
| **Cloud deployment (AWS/GCP)** | Dockerfile | Cloud natively supports Docker |
| **CI/CD pipelines** | Dockerfile | GitHub Actions, GitLab CI use Docker |
| **Local development** | Either | Docker for iteration; Apptainer for final HPC deployment |
| **Sharing with collaborators on HPC** | Apptainer `.sif` | Single file, no setup needed |
| **Sharing with collaborators on cloud** | Dockerfile | Docker Hub is the standard |

### Converting Between the Two

**Docker → Apptainer** (most common workflow):
```bash
# Pull a Docker image and convert to .sif
apptainer pull docker://rocm/pytorch:latest

# Or build from a Dockerfile
apptainer build image.sif docker-daemon://image:latest
```

**Apptainer → Docker** (less common):
```bash
# Export .sif to a directory (sandbox)
apptainer build --sandbox sandbox/ image.sif

# Then build a Dockerfile FROM that sandbox
# (not recommended — better to maintain both separately)
```

> **Reference**: [Apptainer and Docker](https://docs.apptainer.org/en/latest/user_guide/building.html#building-container-images) — official guide on building from Docker sources.

---

## Files in This Lesson

| File | Purpose |
|------|---------|
| `pytorch-rocm.def` | Apptainer definition file (HPC-native) |
| `Dockerfile` | Dockerfile equivalent (for comparison) |
| `submit_container.sh` | SLURM batch script to run training inside the container |

### Container Lifecycle

```
  .def file                 .sif file                  Running container
  (blueprint)               (immutable image)          (process on GPU)
      │                         │                           │
      ▼                         ▼                           ▼
┌───────────┐  apptainer   ┌───────────┐  apptainer   ┌───────────┐
│ Bootstrap │    build      │  Read-    │    exec      │  Your     │
│ %post     │──────────────▶│  Only     │──────────────▶│  Training │
│ %runscript│               │  SquashFS │  --rocm      │  Script   │
└───────────┘               └───────────┘               └───────────┘
                                                        Bind mounts
                                                        /scratch:/scratch
                                                        $PWD:/workspace
```

---

## Step-by-Step: Building a Container

### Step 1 — Load Apptainer on the cluster

```bash
# On the INF0090 cluster
module load apptainer/1.4.1-gcc-11.5.0-linux-rocky9-ivybridge-olpavna

# Verify it works
apptainer --version
```

### Step 2 — Understand the definition file

Open `pytorch-rocm.def`:

```singularity
Bootstrap: docker
From: rocm/pytorch:latest

%post
    pip install lightning torchvision

%runscript
    exec python "$@"
```

**Line by line:**
- `Bootstrap: docker` — pull the base image from Docker Hub
- `From: rocm/pytorch:latest` — use AMD's official PyTorch + ROCm image (includes ROCm runtime + PyTorch)
- `%post` — commands that run **during the build** (like a Dockerfile's `RUN`). Here we add `lightning` and `torchvision` on top of the base.
- `%runscript` — the command that runs when you execute `apptainer exec ... .sif`. `"$@"` passes through any arguments you provide.

> **Reference**: [Apptainer Definition Files](https://docs.apptainer.org/en/latest/user_guide/building.html) — all sections of a `.def` file (`%post`, `%runscript`, `%environment`, `%labels`, etc.)

### Step 3 — Set up a build cache (first time only)

Building pulls a ~5 GB base image. Cache it on `/scratch` to avoid filling `/home`:

```bash
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer/cache
export APPTAINER_TMPDIR=/scratch/$USER/apptainer/tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
```

### Step 4 — Build the container image

```bash
cd 01-apptainer-basics

# This takes a few minutes on first run (downloads ~5 GB base image)
apptainer build pytorch-rocm.sif pytorch-rocm.def
```

This creates `pytorch-rocm.sif` — an **immutable SquashFS image**. You can copy it, share it, and it will always produce the same environment.

> **Docker equivalent**: If you had Docker installed, you could build the same container with `docker build -t pytorch-rocm .` using the included `Dockerfile`. The result is functionally identical, but Docker requires a running daemon and root privileges.

### Step 5 — Test the container interactively

```bash
# Start an interactive shell inside the container
apptainer exec --rocm \
  --bind .:/workspace \
  --pwd /workspace \
  pytorch-rocm.sif \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Flags explained:**
| Flag | What It Does |
|------|-------------|
| `--rocm` | Passes the AMD GPU device(s) into the container |
| `--bind .:/workspace` | Mounts the current directory as `/workspace` inside the container |
| `--pwd /workspace` | Sets the working directory to `/workspace` |
| `exec` | Runs a command inside the container (not the `%runscript`) |

> **Reference**: [Apptainer Exec](https://docs.apptainer.org/en/latest/user_guide/cli/apptainer_exec.html) — full list of flags.

---

## Step-by-Step: Running Training via SLURM

### Step 1 — Examine the SLURM script

Open `submit_container.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=pt-container
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=%j.out

module load apptainer/1.4.1-gcc-11.5.0-linux-rocky9-ivybridge-olpavna

apptainer exec --rocm \
  --bind $PWD:/workspace \
  --pwd /workspace \
  pytorch-rocm.sif \
  python train_lightning.py --devices 1 --strategy auto
```

### Step 2 — Submit the job

```bash
sbatch submit_container.sh

# Watch the job
squeue -u $USER

# Check output when done
cat slurm-*.out
```

### Step 3 — Run interactively (for debugging)

```bash
srun -p gpu --gpus=1 --time=00:10:00 \
  apptainer exec --rocm \
  --bind .:/workspace \
  --pwd /workspace \
  pytorch-rocm.sif \
  python train_lightning.py --devices 1 --strategy auto
```

This runs directly in your terminal (useful for debugging since you see stdout/stderr in real-time).

---

## Key Apptainer Commands Cheat Sheet

| Command | Purpose |
|---------|---------|
| `apptainer build image.sif def.def` | Build a container from a definition file |
| `apptainer pull docker://rocm/pytorch:latest` | Pull a Docker image without building |
| `apptainer exec --rocm image.sif python train.py` | Run a command inside the container |
| `apptainer shell image.sif` | Open an interactive shell inside the container |
| `apptainer inspect image.sif` | View metadata (labels, sections) |

> **Reference**: [Apptainer CLI Reference](https://docs.apptainer.org/en/latest/cli/apptainer.html) — complete command reference.

---

## Practice Questions

1. **Why can't you use Docker directly on most HPC clusters?** Think about the daemon requirement and root access.
2. **What is the difference between `apptainer build` and `apptainer pull`?** When would you use each?
3. **How would you add a custom Python package** (e.g., `mlflow`) to the container? Which section of the `.def` file would you modify?
4. **Why do we use `--bind` instead of Docker-style `-v` volume mounts?** What happens if you forget it?

---

## Further Reading

- [Apptainer Quick Start](https://docs.apptainer.org/en/latest/user_guide/quick_start.html)
- [Apptainer on HPC Clusters](https://docs.apptainer.org/en/latest/admin_guide/admin_quickstart.html)
- [Singularity Container Examples](https://github.com/singularityhub/singularity-hpc)
- [Docker to Apptainer Conversion Guide](https://docs.apptainer.org/en/latest/user_guide/building.html#building-container-images)