# Lesson 2: DDP with PyTorch Lightning

This lesson re-implements the DDP training from Lesson 1 using **PyTorch Lightning** — a framework that eliminates distributed training boilerplate. You'll see how Lightning replaces manual process group initialization, `DistributedSampler`, `DDP` wrapping, and rank-0 logging guards with a clean, declarative interface.

---

## The Problem: Raw DDP Boilerplate

Recall what we had to write manually in `train_ddp.py` (Lesson 1):

```python
# Manual process group init
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Manual sampler
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

# Manual model wrapping
ddp_model = DDP(model, device_ids=[rank])

# Manual epoch shuffle
sampler.set_epoch(epoch)

# Manual rank-0 print guard
if rank == 0:
    print(f"Loss: {loss.item():.4f}")

# Manual cleanup
dist.destroy_process_group()
```

That's ~15 lines of distributed plumbing mixed into your training logic. Lightning removes all of it.

---

## Walking Through the Code

### Step 1: Define the Lightning Module

A `LightningModule` replaces your raw `nn.Module` + training loop with **lifecycle methods** — hooks that Lightning calls at the right time:

```python
class LLMLightning(L.LightningModule):
    def __init__(self, vocab_size=65, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()          # stores vocab_size, lr as self.hparams
        self.model = TineLLM(vocab_size=vocab_size, max_seq_len=64)
        self.loss_fn = nn.CrossEntropyLoss()
        self.vocab_size = vocab_size
```

`save_hyperparameters()` is a Lightning convenience — it saves constructor arguments so the model can be reconstructed from a checkpoint. Without it, you'd have to manually serialize and restore model config.

### Step 2: The Training Step

```python
def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)                           # calls forward()
    loss = self.loss_fn(logits.view(-1, self.vocab_size), y.view(-1))
    self.log("train_loss", loss, prog_bar=True)
    return loss
```

This replaces the entire inner loop body from raw DDP. Notice what's **missing**:
- No `optimizer.zero_grad()` — Lightning does it
- No `loss.backward()` — Lightning does it
- No `optimizer.step()` — Lightning does it
- No `if rank == 0` guard — `self.log()` handles it automatically

### Step 3: The Validation Step

```python
def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = self.loss_fn(logits.view(-1, self.vocab_size), y.view(-1))
    self.log("val_loss", loss, prog_bar=True, sync_dist=True)
```

The `sync_dist=True` flag tells Lightning to **reduce the metric across GPUs** (AllReduce) before logging. Without it, each GPU would log its own local validation loss. This is the Lightning equivalent of manually gathering metrics across ranks.

### Step 4: Configure the Optimizer

```python
def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
```

Lightning calls this method to create the optimizer. You can also return schedulers here.

### Step 5: Data Download with `prepare_data()`

```python
def prepare_data(self):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_path = "shakespeare.txt"
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
```

`prepare_data()` is a special Lightning hook that runs **only on rank 0** and **before** any distributed setup. This replaces our manual `if rank == 0: download() + dist.barrier()` pattern from `dataset.py`. Lightning handles the barrier automatically.

> **Cluster note**: If compute nodes lack internet, pre-copy `shakespeare.txt` into the directory. `prepare_data()` will skip the download if the file exists.

### Step 6: The Trainer — One Object to Rule Them All

```python
trainer = L.Trainer(
    accelerator="gpu",          # use GPUs
    devices=args.devices,       # number of GPUs
    num_nodes=1,                # single node
    strategy=args.strategy,     # "ddp" or "auto" or "fsdp"
    max_epochs=args.max_epochs,
    logger=L.CSVLogger("logs", name="llm_lightning"),
)
trainer.fit(model, train_loader, val_loader)
```

The `Trainer` is where all the distributed magic happens. When `strategy="ddp"`, the Trainer:
1. Detects available GPUs and spawns one process per GPU
2. Initializes `dist.init_process_group` with the NCCL backend
3. Wraps your model in `DDP`
4. Replaces your `DataLoader` sampler with `DistributedSampler`
5. Calls `sampler.set_epoch()` automatically each epoch
6. Handles checkpointing, logging, and graceful shutdown

**Switching to single GPU** is trivial — just set `devices=1` and the strategy auto-switches to `"auto"`:

```python
if args.devices == 1:
    strategy = "auto"
```

---

## Side-by-Side: Raw DDP vs Lightning

| Task | Raw PyTorch (`train_ddp.py`) | Lightning (`train_lightning.py`) |
|------|------------------------------|----------------------------------|
| Process group init | `dist.init_process_group(backend="nccl", ...)` | Automatic via `Trainer(strategy="ddp")` |
| Rank assignment | Read `SLURM_PROCID`, `RANK`, `LOCAL_RANK` env vars | Automatic |
| Model wrapping | `DDP(model, device_ids=[rank])` | Automatic |
| Data sharding | `DistributedSampler(dataset, num_replicas=..., rank=...)` | Automatic (Lightning wraps your DataLoader) |
| Epoch shuffle | `sampler.set_epoch(epoch)` | Automatic |
| Gradient sync | AllReduce in `loss.backward()` | Automatic (DDP wraps model) |
| Rank-0 logging | `if rank == 0: print(...)` | `self.log("name", val)` — only rank 0 logs by default |
| Distributed metrics | Manual AllReduce | `self.log(..., sync_dist=True)` |
| Checkpointing | Manual `torch.save(model.state_dict(), ...)` | `ModelCheckpoint` callback |
| Optimizer step | `optimizer.zero_grad(); loss.backward(); optimizer.step()` | `training_step` returns loss; Lightning handles the rest |
| Cleanup | `dist.destroy_process_group()` | Automatic |
| **Lines of distributed code** | **~30** | **0** |

The training logic is identical — same model, same loss, same optimizer. Lightning just abstracts away the orchestration.

---

## Files

| File | Description |
|------|-------------|
| `train_lightning.py` | Lightning training script with `TineLLM` + `ShakespeareDataset` |
| `submit_lightning.sh` | SLURM batch script (runs in Apptainer `/opt/shared/rocm-pytorch.sif`) |
| `submit_lightning_cuda.sh` | SLURM batch script (runs in Apptainer `/opt/shared/rocm-pytorch.sif`) |

---

## Running

### Via SLURM batch

```bash
cd 02-distributed-training/02-lightning-ddp
sbatch submit_lightning.sh
```

These submit scripts run inside Apptainer using `/opt/shared/rocm-pytorch.sif`.



### Via srun (interactive)

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export TORCH_DISTRIBUTED_INIT_METHOD="env://"

srun --mpi=pmix \
  -p gpu --gpus=2 --ntasks=2 --cpus-per-task=4 --time=00:10:00 \
  apptainer exec --rocm \
    --bind "$PWD:$PWD" --pwd "$PWD" \
    /opt/shared/rocm-pytorch.sif \
    python -u train_lightning.py --devices 2 --strategy ddp
```

### Single GPU (no distributed)

```bash
python train_lightning.py --devices 1
```

---

## Questions

1. What lines of code did Lightning eliminate compared to `train_ddp.py` from Lesson 1?
2. How would you switch from DDP to FSDP in Lightning? (Hint: change one argument)
3. When would you still prefer raw PyTorch DDP over Lightning? (Think: debugging, custom gradient sync, research)
4. What does `sync_dist=True` do in `self.log()`? Why is it needed for validation loss but not training loss?
5. What happens if you call `self.log()` with `prog_bar=False`? Where can you still see the metric?
