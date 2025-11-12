#include <torch/extension.h>

__global__ void relu_forward_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(0.0f, x[i]);
}

__global__ void relu_backward_kernel(const float* __restrict__ gy, const float* __restrict__ x, float* __restrict__ gx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gx[i] = x[i] > 0 ? gy[i] : 0.0f;
}

static inline void cfg(int n, dim3& grid, dim3& block){
    int threads = 256;
    block = dim3(threads);
    grid  = dim3((n + threads - 1) / threads);
}

torch::Tensor relu_forward(torch::Tensor input){
    auto x = input.contiguous();
    auto y = torch::empty_like(x);
    int n = x.numel(); dim3 grid, block; cfg(n, grid, block);
    relu_forward_kernel<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

torch::Tensor relu_backward(torch::Tensor grad_output, torch::Tensor input){
    auto gy = grad_output.contiguous();
    auto x  = input.contiguous();
    auto gx = torch::empty_like(x);
    int n = x.numel(); dim3 grid, block; cfg(n, grid, block);
    relu_backward_kernel<<<grid, block>>>(gy.data_ptr<float>(), x.data_ptr<float>(), gx.data_ptr<float>(), n);
    return gx;
}
