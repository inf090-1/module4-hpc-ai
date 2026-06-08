# 1. Single-GPU Training

This lesson inspects the INF0090 cluster GPU hardware and trains a simple MNIST classifier on a single AMD MI300X. It establishes the baseline before moving to distributed training.

## Learning Objectives

- Inspect GPU topology, VRAM, and compute capabilities using SLURM and Python
- Set up a PyTorch environment with ROCm on the cluster
- Train a CNN classifier on MNIST using a single GPU
- Measure training time and accuracy as a baseline for later scaling comparisons

## Files

- `inspect_cluster.sh` — SLURM script that prints node info, GPU details, and InfiniBand status
- `check_gpu_topology.py` — Python script that verifies PyTorch GPU access and prints topology
- `mnist_single_gpu.py` — MNIST CNN classifier trained on one GPU

## 1. Environment Setup

Before running the training code, you need to set up a Python virtual environment with PyTorch and ROCm support.

```bash
module load python
python -m venv ~/venv-pytorch && source ~/venv-pytorch/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```

## 2. Inspect the Cluster

Submit the inspection script to the GPU partition:

```bash
sbatch inspect_cluster.sh
```

Check the output:

```bash
cat slurm-*.out
```

On the INF090 cluster (`g1` node), you should see:
- **CPU**: AMD EPYC 9474F, 20 cores, 1 socket
- **Memory**: 257 GB
- **GPUs**: 2× AMD Instinct MI300X (device ID 0x74a1), 192 GB VRAM each
- **Storage**: NVMe 2.9 TB Samsung + 150 GB system disk
- **ROCm**: 7.2.0
- **InfiniBand**: May not be available on all nodes

## 3. Verify GPU Access with Python

Run the topology checker on a GPU node:

```bash
sbatch --partition=gpu --gpus=1 --cpus-per-task=4 --time=00:05:00 \
  --output=topo-%j.out --wrap='
  export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
  source ~/venv-pytorch/bin/activate
  python check_gpu_topology.py'
```

Expected output:

```
PyTorch version: 2.12.0+rocm7.2
ROCm version: 7.2.41134-65d174c3e
CUDA available (via HIP): True
Device count: 1
  Device 0: AMD Instinct MI300X
    Compute capability: 9.0
    Total VRAM: 192.0 GB
    SM count: 304
```

## 4. Train MNIST on a Single GPU

### Running via Batch Script (`sbatch`)

```bash
sbatch --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --gpus-per-node=1 --time=00:20:00 --output=mnist-test-%j.out \
  --error=mnist-test-%j.err --wrap='
  export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
  source ~/venv-pytorch/bin/activate
  python -u mnist_single_gpu.py'
```

### Running directly with `srun`

```bash
srun -p gpu --gpus=1 --time=00:20:00 --cpus-per-task=4 \
  bash -lc 'export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:$LD_LIBRARY_PATH; source ~/venv-pytorch/bin/activate; python -u mnist_single_gpu.py'
```

### CLI Parameters

You can adjust the training behavior using the script's built-in command-line arguments:

- `--batch-size`: Input batch size for training (default: 256).
- `--test-batch-size`: Input batch size for testing (default: 512).
- `--epochs`: Number of epochs to train (default: 5).
- `--lr`: Learning rate (default: 1e-3).

**Example using `srun` with arguments:**
```bash
srun -p gpu --gpus=1 --time=00:20:00 --cpus-per-task=4 \
  bash -lc 'export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:$LD_LIBRARY_PATH; source ~/venv-pytorch/bin/activate; python -u mnist_single_gpu.py --epochs 10 --batch-size 128 --lr 0.005'
```

## 5. Expected Output

On the INF090 cluster with 1× MI300X:

```
[mnist] Device: cuda:0 (AMD Instinct MI300X)
[mnist] Epoch 1/5 - train_loss: 0.2484 - train_acc: 0.9279 - val_acc: 0.9742
[mnist] Epoch 2/5 - train_loss: 0.0619 - train_acc: 0.9811 - val_acc: 0.9838
[mnist] Epoch 3/5 - train_loss: 0.0429 - train_acc: 0.9869 - val_acc: 0.9882
[mnist] Epoch 4/5 - train_loss: 0.0310 - train_acc: 0.9901 - val_acc: 0.9842
[mnist] Epoch 5/5 - train_loss: 0.0253 - train_acc: 0.9923 - val_acc: 0.9853
[mnist] Final test accuracy: 98.68%
```

**Note:** Epoch 1 is ~35x slower than subsequent epochs. This is expected — PyTorch compiles HIP kernels on first use (JIT compilation). After the first epoch, all kernels are cached and training runs at full speed (~9s/epoch).

## Questions

1. How much VRAM does the MNIST model use? Is it using the GPU efficiently?
2. What would happen if you changed the model to a much larger architecture? At what point would you need multiple GPUs?
3. Why is it important to establish a single-GPU baseline before moving to distributed training?
