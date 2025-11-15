import time, torch
from torch.utils.cpp_extension import load

relu_native = load(
    name="relu",
    sources=["ops/relu/relu_op.cpp", "ops/relu/relu_kernel.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

class ReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return relu_native.forward(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return relu_native.backward(grad_out.contiguous(), x)

def relu(x):
    return ReLUFn.apply(x)

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
