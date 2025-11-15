# PyTorch CUDA OPs
A curated collection of hand-written CUDA kernels with PyTorch C++ extensions.
This project documents the process of learning and mastering custom CUDA operations, primarily for deep learning frameworks.

Each OP is implemented with: 
- CUDA kernel(.cu)
- C++ launcher
- ATen/Tensor integration
- TORCH_LIBRARY registration
- (later) autograd(Python + C++ backend)
- Unit Tests(forward + backward)

## Implemented Custom OPs
1. Vector Add
    - Simple elementwise op.
    - CUDA -> C++ -> PyTorch dispatcher.
    - Serves as the base template for future ops.
    - Forward-only for now.
2. ReLU
    - Simple elementwise op.
    - Autograd(backward)

## In Progress / Upcoming
- Add autograd
    - Python torch.autograd.Function (as an option for now) 
    - C++-autograd(later) nodes
- ReLU(forward + backward)
- Sigmoid(forward + backward)
- Custom conv2d naive
- im2col + col2im

## Milestone(Exam) OP:
(One of the following or both because why not): 
- Tiled Matmul(shared memory, register tiling)
- Fused Conv + ReLU(kernel fusion)

## Goals 
- Build a high-quality portfolio of CUDA kernels
- Become fluent with PyTorch C++/CUDA extensions
