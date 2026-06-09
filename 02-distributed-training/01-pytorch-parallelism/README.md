# Lesson 1: PyTorch Parallelism Approaches

This lesson walks through the three fundamental parallelism strategies in PyTorch — **Data Parallelism (DDP)**, **Pipeline Parallelism (PP)**, and **Tensor Parallelism (TP)** — by training the same GPT-like language model on the Tiny Shakespeare dataset. We implement each strategy from scratch so you can see exactly what changes and what stays the same.

---

## The Shared Building Blocks

All three training scripts share two files: `model.py` and `dataset.py`.

### The Model — `model.py`

This lesson uses a **small GPT-like Transformer** called **`TineLLM`** so it’s practical to run on just a couple of GPUs.

In plain terms:
- The model reads a sequence of characters and produces a score for **every next character** at every position.
- The Transformer’s **self-attention** lets each position “look at” earlier characters.
- The **causal mask** blocks attention to future characters (so the model can’t cheat during training).
- A final linear layer converts the Transformer’s internal representation back into vocabulary-sized character scores.

The defaults are intentionally small (`d_model=128`, `num_layers=2`) and the output head uses `bias=False` to slightly reduce the number of parameters.

### The Dataset — `dataset.py`

```python
class ShakespeareDataset(Dataset):
    def __init__(self, seq_len=32, train=True):
        ...
        if rank == 0:
            if not os.path.exists(file_path):
                urllib.request.urlretrieve(url, file_path)
        if is_dist:
            dist.barrier()
```

The dataset tokenizes Shakespeare's complete works at the **character level** (~65 unique characters). Each sample is a sliding window of `seq_len` characters, where the target is the same window shifted by one position (next-character prediction).

Only **rank 0** downloads the text file. Other ranks wait at a barrier so they never try to read `shakespeare.txt` before it exists.

---

## 1. Data Parallelism (DDP) — `train_ddp.py`

**Core idea**: Replicate the entire model on every GPU. Split the data across GPUs. Synchronize gradients after each backward pass.

### Step 1: Initialize the Process Group

Every distributed PyTorch program starts by initializing a process group — a communication channel between all GPU processes:

```python
def setup():
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        world_size = int(os.environ["SLURM_NTASKS"])
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
```

Here's what each piece does:

| Variable | Purpose |
|----------|---------|
| `rank` | This process's global ID (0, 1, ..., world_size-1) |
| `local_rank` | Which GPU on this node to use (maps to `cuda:0`, `cuda:1`, ...) |
| `world_size` | Total number of processes across all nodes |
| `MASTER_ADDR` | IP address of the rank-0 process (the coordinator) |
| `MASTER_PORT` | Port for rendezvous communication |
| `backend="nccl"` | NVIDIA/AMD GPU-optimized communication library |

The script detects whether it's running under SLURM or locally. Under SLURM, rank/world_size come from environment variables. Locally, we fall back to `torch.cuda.device_count()` and `mp.spawn` conventions.

### Step 2: Shard the Data with DistributedSampler

```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
loader = DataLoader(dataset, batch_size=64, sampler=sampler)
```

`DistributedSampler` ensures each GPU sees a **different, non-overlapping** subset of the dataset. Without it, every GPU would train on the same batches — wasting compute and producing incorrect gradients.

**Critical**: Call `sampler.set_epoch(epoch)` at the start of each epoch, otherwise every epoch uses the same data ordering:

```python
for epoch in range(2):
    sampler.set_epoch(epoch)  # reshuffle each epoch differently
```

### Step 3: Wrap the Model with DDP

```python
 model = TineLLM(vocab_size=vocab_size, max_seq_len=seq_len).to(rank)
ddp_model = DDP(model, device_ids=[rank])
```

`DDP` wraps your model and **automatically synchronizes gradients** during backward. When `loss.backward()` is called, DDP inserts AllReduce operations into the backward pass so that every GPU ends up with the same averaged gradients. This is the key abstraction — you write the same training loop as single-GPU, and DDP handles the rest.

### Step 4: Standard Training Loop

```python
optimizer.zero_grad()
outputs = ddp_model(inputs)
loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
loss.backward()   # gradients are automatically synchronized here
optimizer.step()
```

The only difference from single-GPU training is using `ddp_model` instead of `model`. The backward pass triggers gradient AllReduce transparently.

### Step 5: Cleanup

```python
dist.destroy_process_group()
```

Always tear down the process group at the end to release NCCL resources.

### How DDP Works Under the Hood

```
GPU 0: forward → loss → backward → AllReduce gradients ─┐
GPU 1: forward → loss → backward → AllReduce gradients ─┤→ same gradients on both GPUs
                                                          └→ optimizer.step()
```

During backward, DDP buckets gradients and overlaps communication with computation. Each parameter's gradient is AllReduced (sum + divide by world_size) so all ranks see identical gradients.

### When to Use DDP

- Your model **fits entirely on one GPU**
- You want the **simplest** distributed approach
- You care most about **throughput** (linear scaling with more GPUs on data-bound workloads)

---

## 2. Pipeline Parallelism (PP) — `train_pp.py`

**Core idea**: Split the model **layer-by-layer** across GPUs. Data flows sequentially from GPU 0 → GPU 1. Each GPU holds only a portion of the model.

### Step 1: Define a Split Model

```python
 class PipelineParallelTineLLM(nn.Module):
     def __init__(
         self,
         vocab_size=65,
         d_model=128,
         nhead=4,
         num_layers=2,
         dim_feedforward=256,
     ):
         super().__init__()
         self.dev0 = "cuda:0"
         self.dev1 = "cuda:1"

        # Part 1 on GPU 0
        self.embedding = nn.Embedding(vocab_size, d_model).to(self.dev0)
        self.pos_encoder = PositionalEncoding(d_model).to(self.dev0)
        self.transformer_part1 = nn.TransformerEncoder(..., num_layers=num_layers//2).to(self.dev0)

        # Part 2 on GPU 1
        self.transformer_part2 = nn.TransformerEncoder(..., num_layers=num_layers//2).to(self.dev1)
         self.linear = nn.Linear(d_model, vocab_size, bias=False).to(self.dev1)
```

Instead of putting the whole model on one device, we **manually place** different layers on different GPUs. The first `num_layers//2` transformer layers go on GPU 0, and the remaining layers + output head go on GPU 1. This requires no `dist.init_process_group` — it's a single process managing two GPUs.

### Step 2: Forward Pass with Explicit Device Transfers

```python
def forward(self, src):
    # GPU 0 forward
    x = self.embedding(src.to(self.dev0))
    x = self.pos_encoder(x)
    x = self.transformer_part1(x, mask=mask_dev0, is_causal=True)

    # Transfer activations from GPU 0 → GPU 1
    x_dev1 = x.to(self.dev1)

    # GPU 1 forward
    x = self.transformer_part2(x_dev1, mask=mask_dev1, is_causal=True)
    output = self.linear(x)
    return output
```

The key line is `x.to(self.dev1)` — this is a **peer-to-peer GPU transfer** (via NCCL/XCCL under the hood) that moves the intermediate activations from GPU 0's memory to GPU 1's memory. PyTorch's autograd tracks this transfer, so gradients will flow back across devices during `loss.backward()`.

### Step 3: Labels Must Be on the Output Device

```python
labels = labels.to("cuda:1")
```

Since the model output lives on GPU 1, the loss computation (which compares output to labels) must also happen on GPU 1. Forgetting this line causes a device mismatch error.

### Step 4: Backward Pass — Automatic Cross-GPU Gradient Flow

```python
loss.backward()
optimizer.step()
```

Even though the model spans two GPUs, `loss.backward()` automatically computes gradients across the device boundary. PyTorch's autograd engine inserts the necessary backward transfers. `optimizer.step()` updates parameters on their respective devices.

### The Pipeline Bubble Problem

This naive implementation processes one batch at a time:

```
Time →  ──────────────────────────────────────────
GPU 0:  [forward part1] .......... [backward part1]
GPU 1:  .................. [forward part2] [backward part2]
                ↑ idle       ↑ idle
```

GPU 0 is **idle** while GPU 1 does its forward pass, and GPU 1 is **idle** while GPU 0 does its forward pass. This is called a **pipeline bubble**. Advanced implementations (GPipe, PipeDream) use **micro-batches** to fill the pipeline:

```
Time →  ──────────────────────────────────────────
GPU 0:  [fwd mb1][fwd mb2][fwd mb3]...[bwd mb1][bwd mb2]...
GPU 1:  ...........[fwd mb1][fwd mb2][fwd mb3]...[bwd mb1]...
```

Our example uses the simple approach for clarity.

### When to Use PP

- Your model is **too large for one GPU** but layers individually fit
- You want to **avoid replicating** the full model (unlike DDP)
- You can tolerate the **pipeline bubble** overhead, or use micro-batching

### PP vs DDP Comparison

| Aspect | DDP | Pipeline Parallelism |
|--------|-----|---------------------|
| Model placement | Full copy on each GPU | Split across GPUs |
| Communication | Gradient AllReduce | Activation transfers (point-to-point) |
| GPU utilization | All GPUs active simultaneously | GPUs idle during pipeline bubbles |
| Memory per GPU | Full model + optimizer | Only model shard + optimizer shard |
| Code complexity | Simple (DDP wrapper) | Manual device placement + transfers |

---

## 3. Tensor Parallelism (TP) — `train_tp.py`

**Core idea**: Split **individual layers** across GPUs. The embedding table, attention matrices, and output projection are partitioned so each GPU holds a fraction of each layer's parameters.

### Step 1: Create a Device Mesh

```python
device_mesh = init_device_mesh("cuda", (world_size,))
```

A `DeviceMesh` is a logical mapping of GPUs. For TP on a single node with 2 GPUs, this creates a 1D mesh: `[GPU 0, GPU 1]`. All tensor parallel operations use this mesh to know which GPU holds which shard.

### Step 2: Define the Parallelization Plan

```python
parallelize_plan = {
    "embedding": RowwiseParallel(input_layouts=None),
    "linear": ColwiseParallel(output_layouts=None),
}
model = parallelize_module(model, device_mesh, parallelize_plan)
```

`parallelize_module` takes a regular `nn.Module` and transforms specified layers into their sharded equivalents:

- **`RowwiseParallel`** on the embedding: Each GPU holds `vocab_size / world_size` rows of the embedding table. With `vocab_size=65` and 2 GPUs, GPU 0 holds 33 rows and GPU 1 holds 32. The output is **partial** — each GPU has a shard of the full embedding output, and an AllReduce is inserted automatically to combine them.

- **`ColwiseParallel`** on the output `nn.Linear`: The weight matrix is split column-wise. GPU 0 holds `d_model × 33` and GPU 1 holds `d_model × 32` columns of the output projection. Each GPU produces a **slice** of the output logits along the vocab dimension.

### Step 3: Compute Loss Across Shards

Each GPU only produces a **slice** of the vocabulary scores, so no single GPU has the full set of logits.

Instead of building the full `(vocab_size)` tensor everywhere, the code computes cross-entropy using distributed math:
- It combines the per-rank parts of the **log-sum-exp denominator** using `all_reduce`.
- It also combines the **target logit** (the logit for the correct next character) across ranks.

Conceptually: *we still get the same cross-entropy loss as single-GPU training, but we never need to materialize the entire vocab logits on every GPU.*

### Why TP Requires High Bandwidth

TP communicates on **every forward and backward pass** — not just during gradient synchronization like DDP. Each sharded layer requires AllReduce or AllGather operations. This is why TP is typically used **within a single node** where GPUs share NVLink/xGMI interconnect (bandwidth ~200+ GB/s), not across nodes where Ethernet/InfiniBand is slower (~25-100 Gb/s).

### When to Use TP

- **Individual layers** are too large for one GPU (e.g., huge embedding tables, massive attention projections)
- You have **high-bandwidth interconnect** between GPUs (NVLink, xGMI)
- You want to reduce per-GPU memory for layers, not just data

### TP vs DDP vs PP Comparison

| Aspect | DDP | PP | TP |
|--------|-----|-----|-----|
| What is split | Data | Model (by layer) | Layers (by tensor) |
| Memory per GPU | Full model | Model shard (layers) | Model shard (tensor slices) |
| Communication | Gradient AllReduce (1x per step) | Activation transfers | AllReduce per sharded layer |
| Bandwidth requirement | Moderate | Low | High |
| GPU utilization | All active | Pipeline bubbles | All active (with overhead) |
| Code complexity | Low (DDP wrapper) | Medium (device placement) | High (mesh, sharding plan, gather) |

---

## Running the Examples

### Via SLURM (cluster: node `g1`)

This lesson includes ready-to-run submit scripts (already configured for `--partition=gpu` and `--nodelist=g1`):

```bash
cd 01-pytorch-parallelism
sbatch submit_ddp.sh
sbatch submit_pp.sh
sbatch submit_tp.sh
```

Each job prints an **epoch summary** like:

```text
Epoch 1/1 | loss: 4.23xx | acc: 0.0x% | time: 12.3s
```

> Logs are written to files like `ddp-<jobid>.out`, `pp-<jobid>.out`, `tp-<jobid>.out`.

#### Container + rendezvous behavior

The submit scripts run the training inside Apptainer (`/opt/shared/rocm-pytorch.sif`).

`submit_ddp.sh` and `submit_tp.sh` set up DDP/TP rendezvous using `env://` (they export `MASTER_ADDR`, `MASTER_PORT`, and `TORCH_DISTRIBUTED_INIT_METHOD=env://`).

`submit_pp.sh` and `submit_tp.sh` additionally load an OpenMPI module because they use `srun --mpi=pmix`.

Optional: you can bypass Apptainer and run with your local venv by setting `USE_VENV=1` when submitting (works for `submit_ddp.sh`, `submit_pp.sh`, and `submit_tp.sh`).

### Locally (2+ GPUs, no SLURM)

```bash
python train_single.py
python train_ddp.py
python train_pp.py
python train_tp.py
```

To compare fairly, run `train_single.py` and the parallel scripts with the same `--seq_len`, `--batch_size`, and `--max_batches`.

> **Cluster note**: Pre-copy `shakespeare.txt` into this directory on the cluster. Compute nodes may lack internet access for auto-download.

---

## Questions

1. In DDP, what happens if you forget `sampler.set_epoch(epoch)`? (Hint: each epoch would see the same data ordering)
2. Why does PP underutilize GPUs compared to DDP? (Think about the pipeline bubble)
3. What communication operations does TP require that DDP does not?
4. Why is TP typically used within a single node rather than across nodes?
5. Could you combine DDP + PP + TP? What would that look like? (Hint: this is how large-scale LLM training works — 3D parallelism)
