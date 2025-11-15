#include <cuda_runtime.h>
#include <cstdint>
#include <c10/cuda/CUDAException.h>

__global__ void relu_forward_kernel(
    const float* __restrict__ x, 
    float* __restrict__ y, 
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(0.0f, x[i]);
}

__global__ void relu_backward_kernel(
    const float* __restrict__ gy, 
    const float* __restrict__ x, 
    float* __restrict__ gx, 
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gx[i] = x[i] > 0 ? gy[i] : 0.0f;
}

void relu_forward_launcher(
    const float* x, 
    float* y, 
    int64_t n
) {
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;

    relu_forward_kernel<<<blocks, threads>>>(x, y, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void relu_backward_launcher(
    const float* gy, 
    const float* x, 
    float* gx,
    int64_t n
) {
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    
    relu_backward_kernel<<<blocks, threads>>>(gy, x, gx, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}