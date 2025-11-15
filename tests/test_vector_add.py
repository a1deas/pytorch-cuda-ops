import os 
import torch
from torch.utils.cpp_extension import load

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "..", "ops", "vector_add")

load(
    name = "pytorch_cuda_ops",
    sources = [
        os.path.join(SRC_DIR, "vector_add.cpp"),
        os.path.join(SRC_DIR, "bindings.cpp"),
        os.path.join(SRC_DIR, "vector_add_kernel.cu"),
    ],
    verbose = True,
    is_python_module=False,
)

def vector_add(a, b):
    return torch.ops.pytorch_cuda_ops.vector_add(a, b)

def main():
    a = torch.rand(1024, device = "cuda", dtype = torch.float32)
    b = torch.rand(1024, device = "cuda", dtype = torch.float32)

    out_custom = vector_add(a, b)
    out_torch = a + b

    max_diff = (out_custom - out_torch).abs().max().item()
    print("max diff:", max_diff)

    assert max_diff < 1e-6
    print("OK: custom CUDA vector_add matches torch add")

if __name__ == "__main__":
    main()