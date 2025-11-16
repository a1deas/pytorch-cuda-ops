# PyTorch CUDA OPs
A curated collection of hand-written CUDA kernels with PyTorch C++ extensions.
This project documents the process of learning and mastering custom CUDA operations, primarily for deep learning frameworks.

Each operation is implemented as a PyTorch C++/CUDA extension with: 
- CUDA kernel(.cu)
- C++ launcher
- ATen/Tensor integration
- TORCH_LIBRARY registration
- (later) autograd(Python + C++ backend)
- Unit Tests(forward + backward)

---

## Implemented Custom OPs
1. Vector Add
    - Simple elementwise op.
    - CUDA -> C++ -> PyTorch dispatcher.
    - Serves as the base template for future ops.
    - Forward-only for now.
    - Benchmark coverage.
2. ReLU
    - Simple elementwise op.
    - Autograd(forward + backward)
    - Benchmark coverage.

## Benchmarks and Tests Results
- GPU: NVIDIA GeForce RTX 5060 Laptop 

1. Vector Add 
```text
n =    10000 | custom vector add:  0.0626 ms | simple add:  0.2722 ms
n =   100000 | custom vector add:  0.0269 ms | simple add:  0.0114 ms
n =  1000000 | custom vector add:  0.0178 ms | simple add:  0.0145 ms
```

2. ReLU
```text
n =    10000 | custom ReLU:  0.0491 ms | torch.relu:  0.4128 ms
n =   100000 | custom ReLU:  0.0198 ms | torch.relu:  0.0248 ms
n =  1000000 | custom ReLU:  0.0199 ms | torch.relu:  0.0123 ms
```

> It is to be expected that a pure learning kernel without tuning will lose out to carefully crafted native functions, but the gap is not that significant.

---

## In Progress / Upcoming
- Sigmoid(forward + backward)
- Custom conv2d naive
- im2col + col2im

## Milestone / "Exam" Ops
(One of the following or both because why not): 
- Tiled Matmul(shared memory, register tiling)
- Fused Conv + ReLU(kernel fusion)

## Goals 
- Build a high-quality portfolio of CUDA kernels
- Become fluent with PyTorch C++/CUDA extensions

## Environment
- PyTorch: 2.9.1
- CUDA: 13.0.1
- Compiler: nvcc, g++
- OS: WSL2(Docker-based workflow)