#include <cuda_runtime.h>
#include <cstdint>
#include <c10/cuda/CUDAException.h>

__global__ void sigmoid_forward_kernel(
    const float* __restrict__ x, 
    float* __restrict__ y,
    int64_t n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 1.0f / (1.0f + expf(-v));
    }
}

__global__ void sigmoid_backward_kernel(
    const float* __restrict__ gy, 
    const float* __restrict__ x,
    float* __restrict__ gx,
    int64_t n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float s = 1.0f / (1.0f + expf(-v));
        gx[i] = gy[i] * s * (1.0f - s);
    }
}

void sigmoid_forward_launcher(
    const float* x, 
    float* y,
    int64_t n 
) {
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    sigmoid_forward_kernel<<<blocks, threads>>>(x, y, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sigmoid_backward_launcher(
    const float* gy, 
    const float* x,
    float* gx,
    int64_t n
) {
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    sigmoid_backward_kernel<<<blocks, threads>>>(gy, x, gx, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}