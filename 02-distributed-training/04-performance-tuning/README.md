# Lesson 4: Performance Tuning & Profiling

This lesson focuses on **finding and fixing performance bottlenecks** in PyTorch training. We cover the three most common optimization targets — batch size, data loading, and memory management — and then show how to use profiling tools to measure the impact. The key insight: **before optimizing, measure**.

---

## Common Bottlenecks in Training

A training loop has three main phases that can bottleneck performance:

```
CPU: [Load batch] → [Transfer to GPU] →              → [Optimizer step]
GPU:                                    [Forward+Backward]
```

1. **Data loading** (CPU): Reading from disk, tokenizing, augmenting — can starve the GPU if the DataLoader can't keep up
2. **Data transfer** (CPU→GPU): Moving tensors from host memory to GPU memory — slow without `pin_memory`
3. **Computation** (GPU): Forward pass, backward pass, optimizer step — the actual work

The goal is to keep the **GPU busy at all times**. Any gap between batches is wasted compute.

---

## Tuning Strategy 1: Maximize Batch Size

```python
train_loader = DataLoader(train_data, batch_size=args.batch_size, ...)
```

Larger batches mean:
- Better GPU utilization (more parallel work per kernel)
- More stable gradient estimates
- But: more VRAM consumed (activations + gradients scale linearly with batch size)

**Rule of thumb**: Increase batch size until you hit ~90% VRAM utilization. Then back off one step.

| Batch Size | GPU Utilization | Memory | Throughput |
|------------|----------------|--------|------------|
| 8 | Low | Low | Low |
| 16 | Medium | Medium | Medium |
| 64 | High | High | High |
| 128 | Near-peak | Near-OOM | Peak |

---

## Tuning Strategy 2: Parallel Data Loading

```python
train_loader = DataLoader(train_data, batch_size=args.batch_size,
                          num_workers=args.num_workers, pin_memory=True)
```

### `num_workers`

By default (`num_workers=0`), data loading happens on the **main thread** — the same thread that runs the GPU computation. This means the GPU **waits** while the CPU loads the next batch:

```
num_workers=0:
CPU: [load batch 1] [idle] [load batch 2] [idle] [load batch 3]
GPU: ...............[compute 1]..........[compute 2]..........[compute 3]
```

With `num_workers=4`, four background processes pre-load batches in parallel:

```
num_workers=4:
CPU workers: [load 1][load 2][load 3][load 4] → [load 5][load 6]...
GPU: .........[compute 1][compute 2][compute 3][compute 4]...
```

The GPU never waits because the next batch is already ready.

**Rule of thumb**: Start with `num_workers=4` per GPU. Increase until throughput plateaus (usually 4-8). Going beyond CPU core count hurts due to context switching.

### `pin_memory=True`

When `pin_memory=False`, data follows this path:

```
Disk → CPU RAM (pageable) → GPU VRAM
                      ↑ slow copy (pageable → pinned → GPU)
```

With `pin_memory=True`:

```
Disk → CPU RAM (pinned) → GPU VRAM
               ↑ fast DMA transfer (pinned → GPU directly)
```

Pinned (page-locked) memory enables **DMA (Direct Memory Access)** transfers — the GPU can pull data directly without CPU involvement. This overlaps data transfer with computation and can be 2-10× faster for host-to-device copies.

---

## Tuning Strategy 3: Memory-Efficient Gradient Reset

```python
optimizer.zero_grad(set_to_none=True)
```

By default, `zero_grad()` fills gradient tensors with zeros — they still occupy memory. With `set_to_none=True`, gradient tensors are set to `None` instead, freeing the memory immediately. PyTorch's backward pass handles `None` gradients by allocating new tensors on demand.

This reduces memory fragmentation and can improve training speed by ~5% in some cases.

---

## Walking Through the Code

### NVTX/ROCTX Markers — Annotating the Profile

The `SimpleLLM` in this lesson is instrumented with **NVTX markers** — named ranges that appear in profiler output:

```python
def forward(self, src):
    torch.cuda.nvtx.range_push("Embedding")
    src = self.embedding(src)
    src = self.pos_encoder(src)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("Transformer")
    seq_len = src.size(1)
    mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
    output = self.transformer_encoder(src, mask=mask, is_causal=True)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("Linear")
    output = self.linear(output)
    torch.cuda.nvtx.range_pop()
    return output
```

`range_push("name")` / `range_pop()` create a **named span** in the profiler timeline. On NVIDIA GPUs, these appear as colored ranges in Nsight Systems. On AMD GPUs (ROCm), PyTorch **automatically maps** `nvtx.range_push` → `roctx.range_push`, so the same code works on both platforms.

### Training Loop Markers

The training loop is also instrumented at a finer granularity:

```python
for batch_idx, (data, target) in enumerate(loader):
    torch.cuda.nvtx.range_push(f"Batch_{batch_idx}")

    torch.cuda.nvtx.range_push("DataTransfer")
    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
    torch.cuda.nvtx.range_pop()

    optimizer.zero_grad(set_to_none=True)

    torch.cuda.nvtx.range_push("ForwardPass")
    output = model(data)
    loss = criterion(output.view(-1, 65), target.view(-1))
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("BackwardPass")
    loss.backward()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("OptimizerStep")
    optimizer.step()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop()  # End Batch
```

| Marker | What it covers | What to look for |
|--------|---------------|-----------------|
| `Embedding` | Embedding lookup + positional encoding | Should be fast — mostly memory access |
| `Transformer` | Self-attention + FFN layers | Dominates compute time |
| `Linear` | Output projection | Small relative to Transformer |
| `DataTransfer` | CPU→GPU copy | Should be tiny with `pin_memory=True` |
| `ForwardPass` | Full forward + loss | Overlaps with Embedding/Transformer/Linear |
| `BackwardPass` | Full backward | Usually 2× forward time |
| `OptimizerStep` | Parameter update | Small but non-zero |
| `Batch_N` | Full iteration | Total time per step |

### `non_blocking=True` on Data Transfer

```python
data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
```

With `non_blocking=True`, the host-to-device transfer happens **asynchronously** — the CPU doesn't wait for the transfer to complete. This allows the CPU to proceed to the next operation (like enqueuing the forward pass) while the data is still in flight. Combined with `pin_memory=True`, this enables true overlap between data transfer and computation.

### Warmup Before Timing

```python
model.train()
dummy_x = torch.randint(0, 65, (args.batch_size, 128)).to(device)
dummy_y = torch.randint(0, 65, (args.batch_size, 128)).to(device)
out = model(dummy_x)
loss = criterion(out.view(-1, 65), dummy_y.view(-1))
loss.backward()
optimizer.step()
torch.cuda.synchronize()
```

GPU operations are **asynchronous** — `model(x)` returns immediately while the GPU works in the background. The first few iterations also trigger **JIT compilation** and **memory allocator warmup**. By running one dummy iteration before timing, we ensure:
1. All kernels are compiled and cached
2. Memory pools are pre-allocated
3. `torch.cuda.synchronize()` waits until the GPU finishes, so our timer starts from a known state

### Timing with Synchronization

```python
start = time.time()
train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
torch.cuda.synchronize()  # wait for GPU to finish
epoch_time = time.time() - start
```

`torch.cuda.synchronize()` is essential — without it, `time.time()` would measure only the time to **enqueue** operations, not the time for the GPU to **complete** them. The GPU runs ahead of the CPU, so the CPU timer would return prematurely.

---

## Profiling Tools: NVIDIA vs AMD

### NVIDIA Nsight Systems (`nsys`)

```bash
nsys profile -t cuda,nvtx,osrt --stats=true -o report python train_tuning.py
```

| Flag | Meaning |
|------|---------|
| `-t cuda` | Trace CUDA API calls and kernel executions |
| `-t nvtx` | Capture NVTX marker ranges |
| `-t osrt` | Trace OS runtime (threads, synchronization) |
| `--stats=true` | Print summary statistics to stdout |
| `-o report` | Save to `report.qdrep` / `report.nsys-rep` |

Open the output in the **Nsight Systems GUI** to see a timeline:

```
Time →  ──────────────────────────────────────────────
CPU Thread 0:  [DataTransfer][ForwardPass][BackwardPass][OptimizerStep]
GPU Kernel:    ..............[embedding][attention][ffn][linear][backward]...
NVTX:          [Batch_0][Batch_1]...
```

You can visually identify:
- **Gaps** between kernels (CPU bottleneck, data loading issues)
- **Long kernels** (which layer dominates)
- **Overlap** between CPU and GPU (good — means async is working)

### AMD `rocprof-sys`

```bash
rocprof-sys --roctx-trace --hip-trace -d output_dir python train_tuning.py
```

| Flag | Meaning |
|------|---------|
| `--roctx-trace` | Capture ROCTX marker ranges (same API as NVTX) |
| `--hip-trace` | Trace HIP kernel executions |
| `-d output_dir` | Save traces to this directory |

Output is in **Perfetto format** (protobuf) — open in Chrome at `chrome://tracing` or use the Perfetto UI at `ui.perfetto.dev`. The timeline view is similar to Nsight Systems.

### Key Difference: Same API, Different Backends

PyTorch's `torch.cuda.nvtx` module is the **unified annotation API**:

```python
torch.cuda.nvtx.range_push("my_marker")   # Works on BOTH NVIDIA and AMD
# ... code to profile ...
torch.cuda.nvtx.range_pop()
```

On NVIDIA: mapped to `nvtxRangePush` / `nvtxRangePop`
On AMD (ROCm): mapped to `roctxRangePush` / `roctxRangePop`

This means your instrumentation code is **portable** — write it once, profile on either platform.

---

## Profiling Tools Comparison

| Feature | NVIDIA (`nsys`) | AMD (`rocprof-sys`) |
|---------|-----------------|---------------------|
| Command | `nsys profile -t cuda,nvtx,osrt --stats=true` | `rocprof-sys --roctx-trace --hip-trace` |
| Output format | `.nsys-rep` / `.qdrep` | Perfetto protobuf |
| Viewer | Nsight Systems (desktop) | Perfetto (Chrome), AMD uProf |
| Kernel tracing | Automatic (CUDA) | Automatic (HIP) |
| Marker API | `torch.cuda.nvtx.range_push/pop` | Same API (mapped to ROCTX) |
| GPU memory tracking | Built-in | Built-in |
| CPU profiling | OS runtime traces | OS runtime traces |
| Multi-GPU support | Yes | Yes |
| Overhead | Low (~5-10%) | Low (~5-10%) |

---

## The Experiments

### Experiment 1: Baseline (slow data loading)

```bash
python train_tuning.py --batch_size 8 --num_workers 0
```

With `num_workers=0`, every batch loads on the main thread. Watch for the GPU sitting idle between steps (visible in profiler as gaps between kernel executions).

### Experiment 2: Optimized loading

```bash
python train_tuning.py --batch_size 16 --num_workers 4
```

With `num_workers=4` and `pin_memory=True`, data loading should no longer bottleneck. The GPU utilization should be much higher — check the `samples/s` metric.

### Experiment 3: Profiling with rocprof-sys (AMD)

```bash
rocprof-sys --roctx-trace --hip-trace -d profile_out python train_tuning.py --batch_size 16 --num_workers 4
```

Open the output in Perfetto UI. Look for:
- NVTX/ROCTX marker ranges matching our code annotations
- Kernel execution timelines (which kernels are longest?)
- Gaps between kernels (signs of CPU bottlenecks)

---

## Files

| File | Description |
|------|-------------|
| `train_tuning.py` | Single-GPU training script with NVTX markers, warmup, and timing |
| `submit_tuning.sh` | SLURM script running all 3 experiments automatically |

---

## Running

### Via SLURM (runs all experiments)

```bash
cd 02-distributed-training/04-performance-tuning
sbatch submit_tuning.sh
```

### Manual runs

```bash
# Baseline
python train_tuning.py --batch_size 8 --num_workers 0 --num_epochs 2

# Optimized
python train_tuning.py --batch_size 16 --num_workers 4 --num_epochs 2

# Profiling (AMD)
rocprof-sys --roctx-trace --hip-trace -d profile_out python train_tuning.py --batch_size 16 --num_workers 4 --num_epochs 1

# Profiling (NVIDIA)
nsys profile -t cuda,nvtx,osrt --stats=true -o report python train_tuning.py --batch_size 16 --num_workers 4 --num_epochs 1
```

---

## Questions

1. Which part of the model takes the longest in the profiler — Embedding, Transformer, or Linear? Why?
2. How does the `DataTransfer` marker duration change when `pin_memory=False` vs `pin_memory=True`?
3. What happens to `samples/s` when you increase `num_workers` from 0 to 4 to 8? Where does it plateau?
4. In the profiler timeline, how do you identify whether the GPU is waiting for data? (Hint: look for gaps between kernel executions)
5. Why do we call `torch.cuda.synchronize()` before measuring time? What would happen without it?
6. What is the effect of `non_blocking=True` on `data.to(device)`? When would it not help?
