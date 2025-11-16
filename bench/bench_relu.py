import os, sys, time, torch
from torch.utils.cpp_extension import load

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "ops", "relu")

load(
    name = "pytorch_cuda_ops",
    sources=[
        os.path.join(SRC_DIR, "relu.cpp"),
        os.path.join(SRC_DIR, "bindings.cpp"),
        os.path.join(SRC_DIR, "relu_kernel.cu"),
    ],
    extra_cflags = ["-O3"],
    extra_cuda_cflags = ["-O3"],
    is_python_module = False,
    verbose = False,
)

from custom.relu import relu as relu_custom

def bench_forward(function, x, iters = 100):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = function(x)
    torch.cuda.synchronize()
    return (time.time() - start) * 1e3 / iters

def main():
    for n in [10000, 100000, 1000000]:
        x = torch.randn(n, device = "cuda", dtype = torch.float32, requires_grad = False)
        t_custom = bench_forward(lambda t: relu_custom(t), x)
        t_torch = bench_forward(lambda t: torch.relu(t), x)
        print(f"n = {n:>8} | custom ReLU: {t_custom:7.4f} ms | torch.relu: {t_torch:7.4f} ms")

if __name__ == "__main__":
    main()