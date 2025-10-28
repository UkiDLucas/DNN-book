import time, statistics as stats
import torch

def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def matmul_bench(device, n=8000, iters=20, warmup=3):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn((n, n), device=device, generator=g)
    y = torch.randn((n, n), device=device, generator=g)
    # warmup
    for _ in range(warmup):
        z = x @ y
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        z = x @ y
        sync()
        times.append(time.perf_counter() - t0)
    return stats.median(times)

def trainstep_bench(device, batch=8192, width=2048, iters=50, warmup=5):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn((batch, width), device=device, generator=g)
    y = torch.randn((batch, 1), device=device, generator=g)
    model = torch.nn.Sequential(
        torch.nn.Linear(width, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, 1),
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    # warmup
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    sync()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        sync()
        times.append(time.perf_counter() - t0)
    return stats.median(times)

def run_all():
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for d in devices:
        torch.set_num_threads(torch.get_num_threads())  # report default
        mm = matmul_bench(d)
        ts = trainstep_bench(d)
        print(f"{d.type.upper():>4}  matmul median: {mm:.3f}s   trainstep median: {ts:.3f}s")

if __name__ == "__main__":
    run_all()