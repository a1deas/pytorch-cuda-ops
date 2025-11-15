#include <torch/extension.h>
#include <cstdint>

void relu_forward_launcher(
    const float* x, 
    float* y, 
    int64_t n
);

void relu_backward_launcher(
    const float* gy, 
    const float* x, 
    float* gx,
    int64_t n
);

at::Tensor relu_forward(const at::Tensor& input){
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "only float32 is supported for now");
    
    auto x = input.contiguous();
    auto y = at::empty_like(x);

    const auto n = x.numel();
    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    relu_forward_launcher(x_ptr, y_ptr, n);
    return y;
}

at::Tensor relu_backward(const at::Tensor& grad_output, const at::Tensor& input){
    TORCH_CHECK(grad_output.device().is_cuda(), "grad_output must be CUDA");
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(grad_output.sizes() == input.sizes(), "grad output and input must have the same shapes");
    TORCH_CHECK(grad_output.dtype() == at::kFloat && input.dtype() == at::kFloat, "only float32 is supported for now");
    
    auto gy = grad_output.contiguous();
    auto x  = input.contiguous();
    auto gx = at::empty_like(x);

    const auto n = x.numel();
    const float* gy_ptr = gy.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    float* gx_ptr = gx.data_ptr<float>();

    relu_backward_launcher(gy_ptr, x_ptr, gx_ptr, n);
    return gx;
}
