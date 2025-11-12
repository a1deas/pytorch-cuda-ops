import time, torch
from torch.utils.cpp_extension import load

relu = load(
    name="relu",
    sources=["ops/relu/relu_op.cpp", "ops/relu/relu_kernel.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

x = torch.randn(1_000_000, device="cuda", dtype=torch.float32, requires_grad=True)

torch.cuda.synchronize(); t0 = time.time()
y = relu.forward(x)
torch.cuda.synchronize(); print("forward ms:", (time.time()-t0)*1e3)

loss = y.sum()
torch.cuda.synchronize(); t0 = time.time()
loss.backward()
torch.cuda.synchronize(); print("backward ms:", (time.time()-t0)*1e3)

print("y[min,max]:", float(y.min()), float(y.max()))
print("grad[min,max]:", float(x.grad.min()), float(x.grad.max()))
