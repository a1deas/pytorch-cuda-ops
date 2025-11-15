#include <torch/extension.h>

at::Tensor relu_forward(const at::Tensor& input);
at::Tensor relu_backward(const at::Tensor& grad_output, const at::Tensor& input);

TORCH_LIBRARY(pytorch_cuda_ops, m) {
    m.def("relu(Tensor input) -> Tensor");
    m.def("relu_backward(Tensor grad_output, Tensor grad_input) -> Tensor");

    m.impl("relu", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(relu_forward)));
    m.impl("relu_backward", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(relu_backward)));
}