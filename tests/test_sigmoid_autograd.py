import os, sys, torch
from torch.utils.cpp_extension import load

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "ops", "sigmoid")

load(
    name = "pytorch_cuda_ops",
    sources = [
        os.path.join(SRC_DIR, "sigmoid.cpp"),
        os.path.join(SRC_DIR, "bindings.cpp"),
        os.path.join(SRC_DIR, "sigmoid_kernel.cu"),
    ],
    extra_cflags = ["-O3"],
    extra_cuda_cflags = ["-O3"],
    is_python_module = False,
    verbose = False,
)

from custom.sigmoid import sigmoid as sigmoid_custom

def main():
    x_ref = torch.randn(128, 64, device = "cuda", dtype = torch.float32, requires_grad = True)
    x_custom = x_ref.detach().clone().requires_grad_(True)

    y_ref = torch.sigmoid(x_ref)
    loss_ref = y_ref.sum()
    loss_ref.backward()
    grad_ref = x_ref.grad.detach().clone()

    y_custom = sigmoid_custom(x_custom)
    loss_custom = y_custom.sum()
    loss_custom.backward()
    grad_custom = x_custom.grad

    max_diff_y = (y_ref - y_custom).abs().max().item()
    max_diff_grad = (grad_ref - grad_custom).abs().max().item()

    print("max diff forward:", max_diff_y)
    print("max diff grad:   ", max_diff_grad)

    assert max_diff_y < 1e-6
    assert max_diff_grad < 1e-6

    print("OK: Sigmoid forward/backward matches torch.sigmoid")

if __name__ == "__main__":
    main()
