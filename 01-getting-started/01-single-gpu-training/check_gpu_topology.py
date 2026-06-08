import torch


def main():
    print(f"PyTorch version: {torch.__version__}")

    hip_version = getattr(torch.version, "hip", None)
    print(f"ROCm version: {hip_version if hip_version else 'N/A (CUDA build)'}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available (via HIP): {cuda_available}")

    if not cuda_available:
        print("No GPU detected. Running on CPU.")
        return

    device_count = torch.cuda.device_count()
    print(f"Device count: {device_count}")

    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        total_mem_gb = props.total_mem / (1024 ** 3)
        print(f"  Device {i}: {name}")
        print(f"    Compute capability: {props.major}.{props.minor}")
        print(f"    Total VRAM: {total_mem_gb:.1f} GB")
        print(f"    SM count: {props.multi_processor_count}")

    device = torch.device("cuda:0")
    x = torch.randn(1000, 1000, device=device)
    y = torch.mm(x, x.T)
    print(f"\nMatrix multiply test: shape={y.shape}, sum={y.sum().item():.2f}")

    allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"VRAM allocated after compute: {allocated:.1f} MB")


if __name__ == "__main__":
    main()
