# 4. MLflow Experiment Tracking & Model Registry

## The Problem

You run dozens of training experiments — different learning rates, batch sizes, model architectures. A week later, you can't remember which combination produced the best result. You check your terminal history, your shell scripts, your notebook outputs... nothing is clearly organized.

**MLflow** solves this by providing a central server where you log **parameters**, **metrics**, **artifacts** (model files, plots), and **code versions** for every experiment run. You can compare runs, register the best model, and deploy it — all from a web UI or Python API.

---

## Learning Objectives

By the end of this lesson you will be able to:
1. Start an MLflow tracking server and log experiments from training code
2. Compare runs in the MLflow web UI
3. Register a model and manage its lifecycle (stages)
4. Load a registered model for inference
5. Integrate MLflow with distributed training (DDP/FSDP)

---

## Software You Will Use

| Tool | What It Does | Where It Lives |
|------|-------------|---------------|
| **MLflow** | Open-source experiment tracking and model registry | Python package (`pip install mlflow`) |
| **MLflow Tracking Server** | Central server storing runs, metrics, artifacts | Login node or dedicated server |
| **SQLite** | Lightweight database for MLflow backend (dev/test) | Local filesystem |
| **PyTorch** | Deep learning framework | Compute node |
| **SLURM** | HPC job scheduler | Cluster |

> **Reference**: [MLflow Documentation](https://mlflow.org/docs/latest/index.html) — official docs covering all features.
> **Reference**: [MLflow Quickstart](https://mlflow.org/docs/latest/getting-started/index.html) — minimal working example.

---

## MLflow Concepts

MLflow consists of four main components that work together:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          MLflow Platform                                 │
│                                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │
│  │   1. Tracking   │  │  2. Projects   │  │   3. Models    │             │
│  │                │  │                │  │                │             │
│  │ Log params,    │  │ Package code   │  │ Save/load in   │             │
│  │ metrics,       │  │ with deps for  │  │ standard       │             │
│  │ artifacts per  │  │ reproducible   │  │ format for     │             │
│  │ run            │  │ runs           │  │ any framework  │             │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘             │
│          │                   │                   │                      │
│          ▼                   ▼                   ▼                      │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │                   4. Model Registry                          │       │
│  │                                                              │       │
│  │  Centralized store for model versions and lifecycle stages   │       │
│  │                                                              │       │
│  │  TinyLLM v1 ──→ Staging ──→ Production ──→ Archived        │       │
│  │  TinyLLM v2 ──→ Staging (current)                           │       │
│  └─────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
```

### How the Components Connect

```
                    Training Script
                         │
                         ▼
              ┌─────────────────────┐
              │  MLflow Tracking    │
              │  (Python API)       │
              │                     │
              │  mlflow.log_param() │
              │  mlflow.log_metric()│
              │  mlflow.log_model() │
              └──────────┬──────────┘
                         │ HTTP/REST
                         ▼
              ┌─────────────────────┐
              │  MLflow Server      │
              │  (Tracking Server)  │
              │                     │
              │  ┌───────────────┐  │
              │  │ Backend Store │  │  ← SQLite (dev) / PostgreSQL (prod)
              │  │ (metadata)    │  │
              │  └───────────────┘  │
              │  ┌───────────────┐  │
              │  │ Artifact Store│  │  ← Local FS / S3 / MinIO
              │  │ (model files) │  │
              │  └───────────────┘  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   MLflow Web UI     │
              │   http://host:5000  │
              │                     │
              │  • Compare runs     │
              │  • View metrics     │
              │  • Download models  │
              │  • Register models  │
              └─────────────────────┘
```

> **Reference**: [MLflow Architecture](https://mlflow.org/docs/latest/tracking.html) — detailed explanation of the tracking server, backend store, and artifact store.

| Concept | What It Is | Example |
|---------|-----------|---------|
| **Experiment** | A logical grouping of related runs | "TinyLLM training" |
| **Run** | A single training execution | One `sbatch` job |
| **Parameter** | Input configuration | `learning_rate=0.001` |
| **Metric** | Measured output | `train_loss=0.42` at step 100 |
| **Artifact** | Saved file | Model weights, plots, configs |
| **Model Registry** | Versioned store for trained models | `TinyLLM/Production` |
| **Model Stage** | Lifecycle status | `None` → `Staging` → `Production` → `Archived` |

> **Reference**: [MLflow Tracking Concepts](https://mlflow.org/docs/latest/tracking.html) — detailed explanation of experiments, runs, params, metrics, artifacts.
> **Reference**: [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html) — model versioning and lifecycle.

---

## Step-by-Step: Setting Up MLflow

### Step 1 — Install MLflow

```bash
# On the cluster, activate your PyTorch environment
source ~/venv-pytorch-rocm/bin/activate

# Install MLflow
pip install mlflow
```

### Step 2 — Start the tracking server

Run this on the **login node** (not a compute node) in a separate terminal:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts
```

**Flags explained:**
| Flag | What It Does |
|------|-------------|
| `--host 0.0.0.0` | Listen on all interfaces (so compute nodes can reach it) |
| `--port 5000` | Port number |
| `--backend-store-uri sqlite:///mlflow.db` | Use SQLite database (simple, no server needed) |
| `--default-artifact-root ./artifacts` | Where to store model files, plots, etc. |

For production, replace SQLite with a real database (PostgreSQL) and use S3/MinIO for artifact storage.

> **Reference**: [MLflow Tracking Server](https://mlflow.org/docs/latest/tracking/server.html) — server configuration options.

### Step 3 — Set the tracking URI

On the **compute node** (before submitting jobs):

```bash
# Replace <login-node> with your login node's hostname or IP
export MLFLOW_TRACKING_URI=http://<login-node>:5000

# Verify connectivity
python -c "import mlflow; print(mlflow.get_tracking_uri())"
# Should print: http://<login-node>:5000
```

### Step 4 — Open the web UI

In your local browser:
```
http://<login-node>:5000
```

You'll see the MLflow dashboard with experiments, runs, and comparisons.

---

## Step-by-Step: Logging an Experiment

### Minimal Example

```python
import mlflow
import torch
import torch.nn as nn

# Connect to the tracking server
mlflow.set_tracking_uri("http://<login-node>:5000")

# Create or retrieve an experiment
mlflow.set_experiment("TinyLLM Training")

# Start a new run
with mlflow.start_run(run_name="lr_0.001_bs_32"):
    # Log parameters
    lr = 0.001
    batch_size = 32
    epochs = 10

    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)

    # Training loop
    model = nn.Linear(128, 65)  # Tiny model for demo
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # ... training code ...
        train_loss = 0.5 * (1 - epoch / epochs)  # Simulated loss

        # Log metrics (with step for time series)
        mlflow.log_metric("train_loss", train_loss, step=epoch)

    # Log the model
    mlflow.pytorch.log_model(model, "model")

    # Log a text artifact
    mlflow.log_text(f"Best loss: {train_loss:.4f}", "summary.txt")

print("Run complete — check MLflow UI!")
```

### What Gets Logged

After running this, in the MLflow UI you'll see:
- **Parameters tab**: `learning_rate=0.001`, `batch_size=32`, `epochs=10`
- **Metrics tab**: `train_loss` plotted over 10 steps (you can zoom, compare)
- **Artifacts tab**: `model/` directory with PyTorch model files, `summary.txt`

---

## Step-by-Step: Distributed Training with MLflow

### The Challenge

In DDP/FSDP, multiple processes run on different GPUs. If all of them log to MLflow simultaneously, you get:
- Duplicate entries (8 copies of every metric)
- File corruption (multiple processes writing the same artifact)
- Server overload

### Solution: Log from Rank 0 Only

```python
import mlflow
import torch.distributed as dist

def setup_mlflow():
    """Initialize MLflow — call once at startup."""
    mlflow.set_tracking_uri("http://<login-node>:5000")
    mlflow.set_experiment("DistributedTraining")

def get_rank():
    """Return the current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0

rank = get_rank()

# Create experiment (all ranks need this to avoid errors)
with mlflow.start_run(run_name=f"ddp_run_{rank}"):
    # Only rank 0 logs parameters and metrics
    if rank == 0:
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("world_size", dist.get_world_size())

    # All ranks train
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader)

        # Only rank 0 logs metrics
        if rank == 0:
            mlflow.log_metric("train_loss", train_loss, step=epoch)

    # Only rank 0 logs the model
    if rank == 0:
        mlflow.pytorch.log_model(model, "model")
```

> **Key point**: `mlflow.start_run()` must be called by all ranks (it's a context manager), but only rank 0 should call `log_param`, `log_metric`, and `log_model`.

### FSDP Model Logging

With FSDP, `model.state_dict()` returns sharded tensors. To log the full model:

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType, FullyShardedDataParallel as FSDP

# Gather full state dict on rank 0
full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
    state_dict = model.state_dict()

if rank == 0:
    # Now you have the full model on rank 0 — safe to log
    mlflow.pytorch.log_model(model, "model")
```

> **Reference**: [PyTorch FSDP State Dict](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullStateDictConfig) — `rank0_only` parameter.

---

## Step-by-Step: Using the Model Registry

The Model Registry lets you version models and manage their lifecycle (e.g., "this model is ready for production").

### Model Registry Lifecycle

```
                    Training Run
                         │
                         ▼
                   ┌───────────┐
                   │  None     │  ← Model just registered
                   │ (v1)      │
                   └─────┬─────┘
                         │ client.transition_model_version_stage("Staging")
                         ▼
                   ┌───────────┐
                   │  Staging  │  ← Testing / validation
                   │ (v1)      │
                   └─────┬─────┘
                         │ client.transition_model_version_stage("Production")
                         ▼
                   ┌───────────┐
                   │Production │  ← Serving live traffic
                   │ (v1)      │
                   └─────┬─────┘
                         │ client.transition_model_version_stage("Archived")
                         ▼
                   ┌───────────┐
                   │ Archived  │  ← Retired / kept for audit
                   │ (v1)      │
                   └───────────┘

  Meanwhile, a new training run produces v2:
                   ┌───────────┐
                   │  None     │  ← v2 registered
                   │ (v2)      │
                   └───────────┘
```

> **Reference**: [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html) — official documentation on model versioning and lifecycle stages.

### Step 1 — Register a model after training

```python
import mlflow
from mlflow import MlflowClient

client = MlflowClient()

# After logging a model in a run, register it
run_id = mlflow.active_run().info.run_id
model_uri = f"runs:/{run_id}/model"

# Register under a model name
model_version = mlflow.register_model(model_uri, "TinyLLM")

print(f"Registered model version: {model_version.version}")
```

### Step 2 — Transition through stages

```python
# Move to "Staging" (for testing)
client.transition_model_version_stage(
    name="TinyLLM",
    version=model_version.version,
    stage="Staging"
)

# After validation, move to "Production"
client.transition_model_version_stage(
    name="TinyLLM",
    version=model_version.version,
    stage="Production"
)
```

### Step 3 — Load a model for inference

```python
# Load by stage
model = mlflow.pytorch.load_model("models:/TinyLLM/Production")

# Use it for prediction
x = torch.randn(1, 128)
with torch.no_grad():
    output = model(x)
print(f"Prediction shape: {output.shape}")
```

### Step 4 — Compare versions in the UI

In the MLflow UI, navigate to **Models → TinyLLM** to see:
- All registered versions
- Which stage each version is in
- The run that produced each version
- Parameters, metrics, and artifacts for each

> **Reference**: [MLflow Model Registry Workflow](https://mlflow.org/docs/latest/model-registry.html#workflows) — detailed lifecycle management.

---

## MLflow Command Reference

| Command | Purpose |
|---------|---------|
| `mlflow server --host 0.0.0.0 --port 5000` | Start tracking server |
| `mlflow experiments create "Experiment Name"` | Create experiment via CLI |
| `mlflow runs list --experiment-id 1` | List runs in an experiment |
| `mlflow ui` | Launch local web UI (for local SQLite) |
| `mlflow models serve -m "models:/Name/Stage"` | Serve model as REST API |

---

## Practice Questions

1. **Why should only rank 0 log to MLflow** in distributed training? What happens if all ranks log?
2. **What is the difference between `mlflow.pytorch.log_model` and `mlflow.pytorch.save_model`?**
3. **How does MLflow handle artifact storage** for large model checkpoints (e.g., 10 GB FSDP shards)?
4. **When would you use the Model Registry** vs just saving model files on disk?
5. **What database would you use in production** instead of SQLite? Why?

---

## Further Reading

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html) — complete reference
- [MLflow Quickstart](https://mlflow.org/docs/latest/getting-started/index.html) — get started in 5 minutes
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) — parameters, metrics, artifacts
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html) — versioning and stages
- [MLflow + PyTorch](https://mlflow.org/docs/latest/models.html#pytorch) — PyTorch integration details
- [MLflow Deployment](https://mlflow.org/docs/latest/models.html#deploying-pytorch-models) — serving models
- [MLflow vs Weights & Biases](https://mlflow.org/docs/latest/getting-started/index.html) — MLflow getting started guide