#include <cuda_runtime.h>
#include <cstdint>
#include <c10/cuda/CUDAException.h>

__global__ void vector_add_kernel(
    const float* __restrict__ a, 
    const float* __restrict__ b, 
    float* __restrict__ out, 
    int64_t n 
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

void vector_add_cuda_launcher(
    const float* a, 
    const float* b, 
    float* out, 
    int64_t n
) {
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;

    vector_add_kernel<<<blocks, threads>>>(a, b, out, n);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}