import os 
import sys
import time 
import torch
from torch.utils.cpp_extension import load

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "ops", "relu")

load(
    name="pytorch_cuda_ops",
    sources=[
        os.path.join(SRC_DIR, "relu.cpp"),
        os.path.join(SRC_DIR, "bindings.cpp"),
        os.path.join(SRC_DIR, "relu_kernel.cu"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    is_python_module = False,
    verbose=True
)

from custom.relu import relu

# simple benchmark
x = torch.randn(1_000_000, device="cuda", dtype=torch.float32, requires_grad=True)

torch.cuda.synchronize(); 
t0 = time.time()
y = relu(x)
torch.cuda.synchronize(); 
print("forward ms:", (time.time()-t0)*1e3)

loss = y.sum()
torch.cuda.synchronize(); 
t0 = time.time()
loss.backward()
torch.cuda.synchronize(); 
print("backward ms:", (time.time()-t0)*1e3)

print("y[min,max]:", float(y.min()), float(y.max()))
print("grad[min,max]:", float(x.grad.min()), float(x.grad.max()))
