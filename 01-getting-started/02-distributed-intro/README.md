# 4. Distributed Training Intro with DDP

This lesson launches your first multi-GPU training job using PyTorch Distributed Data Parallel (DDP) on the INF0090 cluster.

## Learning Objectives

- Understand how DDP distributes training across multiple GPUs
- Launch a multi-GPU training job with `torchrun` via SLURM
- Compare single-GPU vs. multi-GPU training time and throughput

## How DDP Works

When you run `train_ddp.py` with multiple GPUs:

1. **Process spawning**: Each GPU gets its own Python process (via `torchrun` or `srun`)
2. **Process group init**: Processes discover each other via the MASTER address/port
3. **Model replication**: Each GPU holds a full copy of the model
4. **Data sharding**: `DistributedSampler` ensures each GPU sees different data
5. **Gradient sync**: After `loss.backward()`, an AllReduce averages gradients across GPUs
6. **Identical updates**: All replicas stay in sync after `optimizer.step()`

## Files

- `train_ddp.py` — DDP training script with SLURM and torchrun support
- `submit_ddp.sh` — SLURM batch script for 2-GPU training

## Environment Setup

Before running the training code, ensure you have set up a Python virtual environment with PyTorch and ROCm support:

```bash
module load python
python -m venv ~/venv-pytorch && source ~/venv-pytorch/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```

## Running

### Via SLURM batch

```bash
cd 01-getting-started/04-distributed-intro
sbatch submit_ddp.sh
```

Monitor the output:

```bash
tail -f ddp-*.out
```

### Via srun (interactive)

```bash
srun -p gpu --gpus=2 --ntasks=2 --cpus-per-task=4 --time=00:15:00 \
  bash -lc 'export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:$LD_LIBRARY_PATH; source ~/venv-pytorch/bin/activate; python train_ddp.py'
```

### CLI Parameters

You can adjust the training behavior using the script's built-in command-line arguments:

- `--batch-size`: Input batch size for training (default: 256).
- `--test-batch-size`: Input batch size for testing (default: 512).
- `--epochs`: Number of epochs to train (default: 5).
- `--lr`: Learning rate (default: 1e-3).

**Example using `srun` with arguments:**
```bash
srun -p gpu --gpus=2 --ntasks=2 --cpus-per-task=4 --time=00:15:00 \
  bash -lc 'export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:$LD_LIBRARY_PATH; source ~/venv-pytorch/bin/activate; python train_ddp.py --epochs 10 --batch-size 128 --lr 0.005'
```

## Expected Output

On the INF090 cluster with 2× MI300X:

```
=== DDP MNIST Training ===
Nodes: 1
Tasks: 2
GPUs per node: 2
Job ID: 1790

[ddp] World size: 2, Device: cuda:0 (AMD Instinct MI300X)
[ddp] Epoch 1/5 - train_loss: 0.3918 - train_acc: 0.8892 - val_acc: 0.9650
[ddp] Epoch 2/5 - train_loss: 0.0844 - train_acc: 0.9751 - val_acc: 0.9796
[ddp] Epoch 3/5 - train_loss: 0.0597 - train_acc: 0.9819 - val_acc: 0.9848
[ddp] Epoch 4/5 - train_loss: 0.0441 - train_acc: 0.9867 - val_acc: 0.9878
[ddp] Epoch 5/5 - train_loss: 0.0361 - train_acc: 0.9889 - val_acc: 0.9888
[ddp] Final test accuracy: 98.69%
```

**Note:** Epoch 1 is slower due to JIT compilation + DDP initialization overhead. Subsequent epochs stabilize at ~15s.

## Comparison with Single-GPU

Measured on INF090 node `g1` (2× MI300X):

| Metric | Single GPU | 2-GPU DDP | Speedup |
|--------|------------|-----------|---------|
| Epoch 2-5 time | ~9s | ~15s | — |
| Total time | 400.5s | 108.7s | **3.7×** |
| Test accuracy | 98.68% | 98.69% | same |

The total speedup (3.7×) exceeds the per-epoch ratio because:
- Single-GPU epoch 1 takes 359s (JIT compilation), DDP epoch 1 takes 39.4s (compiled on both GPUs in parallel)
- After compilation, per-epoch speedup is ~0.6× (DDP has AllReduce overhead for this small model)
- For larger models with more computation per layer, the speedup approaches the theoretical 2× for 2 GPUs

## Questions

1. Why doesn't 2 GPUs give exactly 2x speedup?
2. What happens to the speedup as you increase the number of GPUs to 4?
3. How would you modify this script to use FSDP instead of DDP?
