#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%j.out

echo "=== Node Info ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

echo "=== CPU Info ==="
lscpu | grep -E 'Model name|Socket|Core|Thread|CPU\(s\):'
echo ""

echo "=== Memory ==="
free -h
echo ""

echo "=== Block Devices ==="
lsblk -d -o NAME,SIZE,TYPE,MODEL
echo ""

echo "=== GPU Info (rocm-smi) ==="
rocm-smi --showid
echo ""
rocm-smi --showmeminfo vram
echo ""
rocm-smi --showtemp
echo ""
rocm-smi --showpower
echo ""

echo "=== InfiniBand ==="
if command -v ibstat &> /dev/null; then
    ibstat
else
    echo "ibstat not found, trying ibv_devinfo..."
    ibv_devinfo 2>/dev/null || echo "No InfiniBand tools available"
fi
echo ""

echo "=== ROCm Version ==="
if [ -f /opt/rocm/.info/version ]; then
    cat /opt/rocm/.info/version
else
    rocm-smi --showproductname 2>/dev/null || echo "Could not determine ROCm version"
fi
echo ""

echo "=== Done ==="
