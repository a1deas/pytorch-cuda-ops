#include <torch/extension.h>

void vector_add_cuda_launcher(
    const float* a,
    const float* b,
    float* out,
    int64_t n
);

at::Tensor vector_add_forward(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == b.dtype(), "a and b must have the same dtype");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a and b must be contiguous for now");
    TORCH_CHECK(a.dtype() == at::kFloat, "only float32(torch.float32) is supported for now");

    auto out = at::empty_like(a);

    const auto n = a.numel();
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    vector_add_cuda_launcher(a_ptr, b_ptr, out_ptr, n);

    return out;
}