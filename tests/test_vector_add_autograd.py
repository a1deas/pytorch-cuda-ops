import os
import sys
import torch
from torch.utils.cpp_extension import load

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "ops", "vector_add")

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

from custom.vector_add import vector_add # our custom function

def main():
    a = torch.randn(5, device = "cuda", dtype = torch.float32, requires_grad = True)
    b = torch.randn(5, device = "cuda", dtype = torch.float32, requires_grad = True)

    # Simple Vector Add addition
    y_ref = (a + b).sum()
    y_ref.backward()
    a_grad_reference = a.grad.clone()
    b_grad_reference = b.grad.clone()
    a.grad.zero_()
    b.grad.zero_()

    # Custom Vector Add realization
    y_custom = vector_add(a, b).sum()
    y_custom.backward()
    a_grad_custom = a.grad
    b_grad_custom = b.grad

    print("grad a reference: ", a_grad_reference)
    print("grad a custom: ", a_grad_custom)
    print("grad b reference:", b_grad_reference)
    print("grad b custom: ", b_grad_custom)

    assert torch.allclose(a_grad_reference, a_grad_custom)
    assert torch.allclose(b_grad_reference, b_grad_custom)

    print("OK: autograd for vector_add matches PyTorch add")

if __name__ == "__main__":
    main()



