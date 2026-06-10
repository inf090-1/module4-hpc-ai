# Lesson 3: FSDP — Fully Sharded Data Parallel

This lesson demonstrates a critical limitation of DDP — **it replicates the entire model on every GPU** — and shows how FSDP solves this by sharding model parameters, gradients, and optimizer states across GPUs. We use a deliberately oversized model to trigger an Out-Of-Memory (OOM) error with DDP, then recover with FSDP.

---

## The Problem: DDP's Memory Scaling

In DDP (Lessons 1 & 2), every GPU holds a **complete copy** of the model, gradients, and optimizer state:

```
GPU 0: [full model params] [full gradients] [full optimizer state]
GPU 1: [full model params] [full gradients] [full optimizer state]
```

If your model has 4 billion parameters (16 GB in fp32), each GPU needs ~64 GB (params + gradients + Adam optimizer state: 2 momentum tensors). This works for small models but breaks when the model exceeds a single GPU's VRAM — **adding more GPUs with DDP doesn't help** because each GPU still needs the full model.

---

## The MassiveModel

To demonstrate this, we create a model that intentionally exceeds a single MI300X's 192 GB VRAM:

```python
class MassiveModel(nn.Module):
    def __init__(self, d_model=4096, num_layers=12, vocab_size=65):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=16, dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)
```

Let's estimate the parameter count:

| Component | Parameters |
|-----------|-----------|
| Embedding: 65 × 4096 | 266K |
| Per layer: self-attn (4 × 4096²) + FFN (4096 × 16384 × 2) + norms | ~100M |
| 12 layers × ~100M | ~1.2B |
| Output head: 4096 × 65 | 266K |
| **Total** | **~1.5B** |

At fp32 with Adam (params + grads + 2 momentum tensors), that's ~24 GB — fits on MI300X. But the `dim_feedforward=d_model*4=16384` creates large intermediate activations. With a batch size of 8 and sequence length 128, activation memory can push well beyond what DDP can handle per-GPU when combined with optimizer states and gradients.

---

## How FSDP Works

FSDP shards the model's **parameters**, **gradients**, and **optimizer state** across GPUs:

```
DDP (2 GPUs):
GPU 0: [full params] [full grads] [full optimizer]
GPU 1: [full params] [full grads] [full optimizer]

FSDP (2 GPUs):
GPU 0: [shard 0 of params] [shard 0 of grads] [shard 0 of optimizer]
GPU 1: [shard 1 of params] [shard 1 of grads] [shard 1 of optimizer]
```

Each GPU stores only **1/N** of the total model state. Before a layer runs its forward pass, FSDP **AllGathers** the full parameters from all shards, computes the layer, then **discards** the gathered parameters (keeping only the local shard). After backward, gradients are **ReduceScattered** so each GPU keeps only its own shard's gradient.

This means FSDP's peak memory ≈ `1/N of model + one full layer (during AllGather)`, vs DDP's `full model`.

### The FSDP Communication Pattern

```
Forward:
  Layer 0: AllGather params → compute → discard gathered params
  Layer 1: AllGather params → compute → discard gathered params
  ...

Backward:
  Layer N: AllGather params → compute grad → ReduceScatter grads → discard
  Layer N-1: AllGather params → compute grad → ReduceScatter grads → discard
  ...
```

The key insight: **FSDP trades memory for communication**. Every layer requires an AllGather (forward) and a ReduceScatter (backward), but in exchange, each GPU only stores 1/N of the model.

---

## Walking Through the Code

### Step 1: The Lightning Module

```python
class MassiveLitModel(L.LightningModule):
    def __init__(self, d_model=4096, num_layers=12, vocab_size=65, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = MassiveModel(d_model=d_model, num_layers=num_layers, vocab_size=vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.vocab_size = vocab_size
```

This is nearly identical to the Lightning module from Lesson 2 — same `training_step`, same `configure_optimizers`. The only difference is the model size.

### Step 2: Monitoring VRAM at Training Start

```python
def on_train_start(self):
    if self.global_rank == 0:
        device = self.device
        vmb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        print(f"\n[massive] VRAM allocated at start: {vmb:.1f} MB on rank {self.global_rank}")
        print(f"[massive] Model structure:\n{self.model}\n")
```

`on_train_start()` is a Lightning hook that fires after the model is moved to GPU but before training begins. We print VRAM usage to show how much memory FSDP saved compared to DDP. `self.global_rank` gives the process's rank across all nodes — only rank 0 prints to avoid duplicated output.

When using FSDP, you'll see `FullyShardedDataParallel` wrappers in the model structure output, showing which layers FSDP is sharding.

### Step 3: The Strategy Switch — DDP vs FSDP

```python
trainer = L.Trainer(
    accelerator="gpu",
    devices=args.devices,
    num_nodes=1,
    strategy=args.strategy,    # "ddp" or "fsdp"
    max_epochs=args.max_epochs,
    logger=L.CSVLogger("logs", name="massive_model"),
)
```

Switching from DDP to FSDP is a **single argument change**: `strategy="fsdp"` instead of `strategy="ddp"`. Lightning handles all the sharding, AllGather, and ReduceScatter logic internally. This is one of Lightning's biggest wins — the same training code works for both strategies.

---

## The Two Scenarios

### Scenario 1: DDP OOM (`submit_oom.sh`)

```bash
python train_fsdp.py --devices 1 --strategy ddp --d_model 4096 --num_layers 12
```

With `--devices 1`, Lightning places the full model on a single GPU. The `MassiveModel` with its large intermediate activations and optimizer states should trigger a CUDA Out-Of-Memory error. This demonstrates why DDP doesn't help with large models — every GPU needs the full copy.

### Scenario 2: FSDP Success (`submit_fsdp.sh`)

```bash
python train_fsdp.py --devices 2 --strategy fsdp --d_model 4096 --num_layers 12
```

With `--devices 2 --strategy fsdp`, the model is sharded across both GPUs. Each GPU stores approximately half the parameters, gradients, and optimizer state. The model that was too large for one GPU now fits across two.

---

## What to Observe

When you run both scenarios, compare:

1. **Scenario 1 output**: You should see an OOM error — `torch.cuda.OutOfMemoryError: CUDA out of memory`
2. **Scenario 2 output**: Training proceeds normally. Look for:
   - **VRAM allocation** in `on_train_start()`: Much lower than the full model size
   - **FSDP wrappers** in the model structure: Layers wrapped in `FullyShardedDataParallel`
   - **Decreasing loss**: Training is actually learning

---

## DDP vs FSDP: Complete Comparison

| Aspect | DDP | FSDP |
|--------|-----|------|
| Model weights | Full copy on every GPU | Sharded across GPUs (1/N per GPU) |
| Gradients | Full copy on every GPU | Sharded (ReduceScatter) |
| Optimizer state | Full copy on every GPU | Sharded (1/N per GPU) |
| Memory per GPU | O(model_size) | O(model_size / N) |
| Communication per step | 1× AllReduce (gradients) | 2× AllGather + 1× ReduceScatter per layer |
| Bandwidth cost | Low | Medium-High |
| Max model size | Must fit on 1 GPU | Can exceed 1 GPU |
| Code change (Lightning) | `strategy="ddp"` | `strategy="fsdp"` |

### When to Use Which

| Situation | Recommended Strategy |
|-----------|---------------------|
| Model fits on 1 GPU, want throughput | DDP |
| Model doesn't fit on 1 GPU | FSDP |
| Model barely fits, want larger batch size | FSDP (frees memory for activations) |
| Multi-node with slow interconnect | DDP (less communication) |
| Multi-node with fast interconnect | FSDP (sharding across nodes) |

---

## Running

The `submit_oom.sh` and `submit_fsdp.sh` scripts run inside Apptainer (`/opt/shared/rocm-pytorch.sif`).



### Scenario 1: OOM Demo (DDP, 1 GPU)

```bash
 sbatch submit_oom.sh
```

Expected: `torch.cuda.OutOfMemoryError`.

Note: `submit_oom.sh` passes larger `--seq_len`, `--batch_size`, and `--ff_mult` values so the single-GPU run actually exceeds the memory budget and triggers OOM.

### Scenario 2: FSDP (2 GPUs, sharded)

```bash
sbatch submit_fsdp.sh
```

Expected: Training completes with decreasing loss.

### Manual run

```bash
# OOM demo
python train_fsdp.py --devices 1 --strategy ddp --d_model 4096 --num_layers 12

# FSDP
python train_fsdp.py --devices 2 --strategy fsdp --d_model 4096 --num_layers 12

# Smaller model that fits with DDP
python train_fsdp.py --devices 2 --strategy ddp --d_model 512 --num_layers 4
```

---

## Questions

1. How much VRAM does each GPU use with DDP vs. FSDP? Check the `[massive] VRAM allocated` output.
2. Why does FSDP have higher communication overhead than DDP? Count the collective operations per layer.
3. When would you choose DDP over FSDP even if the model fits in memory with both?
4. What happens if you run FSDP with `--devices 1`? Does it still work? (Try it — FSDP with 1 device is essentially no sharding)
5. How does FSDP's `ShardingStrategy` parameter affect the trade-off? (Research `FULL_SHARD`, `SHARD_GRAD_OP`, `NO_SHARD`)
