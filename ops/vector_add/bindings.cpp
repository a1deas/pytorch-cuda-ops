#include <torch/extension.h>

at::Tensor vector_add_forward(const at::Tensor& a, const at::Tensor& b);

TORCH_LIBRARY(pytorch_cuda_ops, m) {
    m.def("vector_add(Tensor a, Tensor b) -> Tensor");
    m.impl("vector_add", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(vector_add_forward)));
}