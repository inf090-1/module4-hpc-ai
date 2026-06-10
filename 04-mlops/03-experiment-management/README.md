# 3. Distributed Checkpoint Management

## The Problem

When you train a large model across multiple GPUs, the model's parameters (weights) are split across devices. If the job crashes — due to a hardware fault, time limit, or power outage — you lose all your progress unless you **checkpoint** (save) the model periodically.

But checkpointing in distributed training is tricky: each GPU only holds a **shard** of the model. Simply having each GPU save independently produces corrupt or incomplete files. This lesson teaches you the correct patterns for checkpointing with DDP and FSDP.

---

## Learning Objectives

By the end of this lesson you will be able to:
1. Save and load model checkpoints in a DDP (Data Parallel) setting
2. Save and load checkpoints in an FSDP (Fully Sharded Data Parallel) setting
3. Choose the right checkpoint strategy for your workload
4. Store checkpoints efficiently on HPC parallel filesystems

---

## Software You Will Use

| Tool | What It Does |
|------|-------------|
| **PyTorch `torch.save` / `torch.load`** | Serialize/deserialize Python objects (state dicts, optimizers) |
| **`torch.distributed.checkpoint` (DCP)** | Distributed checkpoint format that handles sharded state |
| **SLURM** | Job scheduler — controls when and where jobs run |
| **Lustre / BeeGFS** | Parallel filesystem — high-bandwidth storage shared across nodes |

> **Reference**: [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html) — official documentation for the DCP API.
> **Reference**: [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) — PyTorch FSDP tutorial with checkpointing examples.

---

## Checkpoint Challenges by Strategy

| Strategy | What's on Each GPU | Challenge | Solution |
|----------|-------------------|-----------|----------|
| **DDP** | Full model replica | Every rank writes → file corruption | Only rank 0 saves |
| **FSDP** | Sharded model (params split) | Each rank has a different shard | Use DCP: all ranks participate |
| **Multi-node** | Shards + activations | Parallel FS contention | Write to `/scratch` (high bandwidth) |

### How FSDP Checkpointing Works

In FSDP, parameters are **sharded** across GPUs during training. Before forward/backward, they are **all-gathered** into full parameters; after computation, they are **resharded**.

![FSDP workflow](https://docs.pytorch.org/tutorials/_images/fsdp_workflow.png)

*Source: [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)*

The same principle applies to checkpointing:

```
Training:  GPU 0 has shard_0  │  GPU 1 has shard_1  │  GPU 2 has shard_2  │  GPU 3 has shard_3
                                      │
                                      ▼
Checkpoint save (DCP):         Each rank saves its shard to a separate .distcp file
                                      │
                                      ▼
Checkpoint directory:          ├── __0_0.distcp   (GPU 0's shard)
                               ├── __1_0.distcp   (GPU 1's shard)
                               ├── __2_0.distcp   (GPU 2's shard)
                               ├── __3_0.distcp   (GPU 3's shard)
                               ├── .metadata
                               └── meta.pt
                                      │
                                      ▼
Checkpoint load (DCP):         Each rank reads its shard → reshards to current topology
```

> **Reference**: [PyTorch FSDP State Dict](https://pytorch.org/docs/stable/fsdp.html#types-of-state-dict) — `FULL_STATE_DICT` vs `SHARDED_STATE_DICT` vs `LOCAL_STATE_DICT`.

---

## Step-by-Step: DDP Checkpointing

In DDP, each GPU holds a **complete copy** of the model. Only one rank needs to save.

### Step 1 — Understand the state dict

```python
# A state dict is a Python dictionary mapping layer names to tensors
print(model.state_dict().keys())
# odict_keys(['encoder.layer1.weight', 'encoder.layer1.bias', 'decoder.fc.weight', ...])
```

### Step 2 — Save checkpoint (rank 0 only)

```python
import torch
import torch.distributed as dist

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pt"):
    """Save checkpoint — call only on rank 0."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return  # Only rank 0 writes

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"Checkpoint saved to {path}")
```

**Why rank 0 only?** In DDP, all ranks have identical model copies. If all 8 GPUs try to write the same file simultaneously, you get data corruption. Only rank 0 writes; other ranks skip this step.

### Step 3 — Load checkpoint

```python
def load_checkpoint(model, optimizer, path="checkpoint.pt", device="cuda"):
    """Load checkpoint and resume training."""
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Resumed from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
    return checkpoint["epoch"], checkpoint["loss"]
```

### Step 4 — Save at regular intervals

```python
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)

    # Save every 5 epochs (or whenever validation improves)
    if epoch % 5 == 0:
        save_checkpoint(model, optimizer, epoch, train_loss, f"ckpt_epoch{epoch}.pt")

    # Also save a "latest" checkpoint (overwritten each time)
    save_checkpoint(model, optimizer, epoch, train_loss, "ckpt_latest.pt")
```

---

## Step-by-Step: FSDP Checkpointing

In FSDP, the model is **sharded** — each GPU holds a different slice of the parameters. You can't just save on rank 0 because rank 0 doesn't have the full model anymore.

### Step 1 — Use the Distributed Checkpoint (DCP) API

```python
from torch.distributed.checkpoint import save, load
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Save — all ranks participate
save(model.state_dict(), checkpoint_dir="checkpoints/epoch_5/")
```

This saves each rank's shard to a separate file in `checkpoints/epoch_5/`. The DCP API coordinates the writing automatically.

### Step 2 — Load from DCP format

```python
# Prepare the state dict (must match the sharded structure)
state_dict = {"model": model.state_dict()}

# Load — all ranks read their respective shards
load(state_dict, checkpoint_dir="checkpoints/epoch_5/")

# Apply loaded weights
model.load_state_dict(state_dict["model"])
```

### Step 3 — Gather full state dict for logging (optional)

Sometimes you need the full model on rank 0 (e.g., to upload to a model registry). Use `FSDP.state_dict_type`:

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
    state_dict = model.state_dict()

if rank == 0:
    torch.save(state_dict, "full_model.pt")  # Now safe to save full model
```

This **gathers** all shards to rank 0, which temporarily uses more CPU memory but gives you the complete model.

> **Reference**: [PyTorch FSDP State Dict Types](https://pytorch.org/docs/stable/fsdp.html#types-of-state-dict) — `FULL_STATE_DICT` vs `SHARDED_STATE_DICT` vs `LOCAL_STATE_DICT`.

---

## Step-by-Step: Running the Example

### Files

| File | Purpose |
|------|---------|
| `train_checkpoint.py` | Training loop with checkpoint save/load |
| `submit_checkpoint.sh` | SLURM batch script |

### Step 1 — Navigate to the lesson directory

```bash
cd 04-mlops/03-experiment-management
```

### Step 2 — Submit the training job

```bash
sbatch submit_checkpoint.sh
```

### Step 3 — Monitor progress

```bash
# Check job status
squeue -u $USER

# Watch output in real-time
tail -f slurm-*.out
```

### Step 4 — Verify checkpoint was saved

```bash
ls -la *.pt
# Should show checkpoint.pt and/or ckpt_latest.pt

# Check checkpoint contents
python -c "
import torch
ckpt = torch.load('checkpoint.pt', map_location='cpu')
print('Keys:', list(ckpt.keys()))
print('Epoch:', ckpt['epoch'])
print('Loss:', ckpt['loss'])
"
```

---

## Where to Store Checkpoints on HPC

| Filesystem | Path | Bandwidth | Persistence | Notes |
|-----------|------|-----------|-------------|-------|
| **Home** | `/home/$USER` | Low | Permanent | Don't store checkpoints here |
| **Scratch** | `/scratch/$USER` | High | 30-60 day auto-delete | **Best for checkpoints** |
| **Local** | `/tmp` | Very high | Lost on reboot | Use only for temporary checkpoints |

**Rule of thumb**: Save to `/scratch`, copy to `/home` or archive when the job finishes.

```bash
# In your SLURM script
CHECKPOINT_DIR="/scratch/$USER/experiments/exp_$(date +%Y%m%d_%H%M%S)"
mkdir -p $CHECKPOINT_DIR

# Save checkpoint
python train.py --save-dir $CHECKPOINT_DIR

# After training, archive important checkpoints
cp $CHECKPOINT_DIR/best_model.pt /home/$USER/archived/
```

> **Reference**: [Lustre Filesystem](https://lustre.readthedocs.io/) — parallel filesystem used on many HPC clusters.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Save only on rank 0 (DDP) | Avoid file corruption from concurrent writes |
| Use DCP for FSDP | Handles sharded state correctly |
| Save every N epochs | Recover from crashes without losing too much work |
| Save "latest" + "best" | Latest for resumption; best for deployment |
| Include optimizer + scheduler state | Resume training without re-warmup |
| Save hyperparameters | Reproduce experiments later |
| Store on `/scratch` | High bandwidth; don't pollute `/home` |
| Archive to permanent storage | `/scratch` auto-deletes after 30-60 days |

---

## Practice Questions

1. **Why can't all ranks save their model simultaneously in DDP?** What would the file look like?
2. **How does FSDP checkpointing differ from DDP?** Why does FSDP need a special API?
3. **What is the advantage of saving to `/scratch` instead of `/home`?** What happens if you forget?
4. **When would you use `FULL_STATE_DICT` vs `SHARDED_STATE_DICT` in FSDP?**
5. **Why should you save the optimizer state?** What happens if you only save the model weights?

---

## Further Reading

- [PyTorch Checkpoint Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#saving-and-loading-checkpoints)
- [FSDP Checkpoint Example](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#saving-and-loading-a-checkpoint)
- [Distributed Checkpoint API](https://pytorch.org/docs/stable/distributed.checkpoint.html)
- [Torch distributed.elastic](https://pytorch.org/docs/stable/elastic.html) — fault-tolerant training
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html) — HPC job scheduling