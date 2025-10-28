import torch, time

device_cpu = torch.device("cpu")
device_mps = torch.device("mps")

def benchmark(device):
    x = torch.rand((5000, 5000), device=device)
    y = torch.rand((5000, 5000), device=device)
    if device.type == "mps":
        torch.mps.synchronize()
    start = time.time()
    z = x @ y
    if device.type == "mps":
        torch.mps.synchronize()
    return time.time() - start

t_cpu = benchmark(device_cpu)
t_mps = benchmark(device_mps)

print(f"CPU time: {t_cpu:.3f} s  |  Metal (MPS) time: {t_mps:.3f} s")

# expected: Torch_acceleration.py"
# macOS M1 64Gb RAM: 
# CPU time: 0.128 s  |  Metal (MPS) time: 0.176 s
# gaming ubuntu: 
# CPU time: 0.333 s  |  GPU time: 0.039 s