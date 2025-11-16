#include <torch/extension.h>

at::Tensor sigmoid_forward(const at::Tensor& input);
at::Tensor sigmoid_backward(const at::Tensor& grad_output, const at::Tensor& input);

TORCH_LIBRARY(pytorch_cuda_ops, m) {
    m.def("sigmoid(Tensor input) -> Tensor");
    m.def("sigmoid_backward(Tensor grad_output, Tensor input) -> Tensor");

    m.impl("sigmoid", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(sigmoid_forward)));
    m.impl("sigmoid_backward", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(sigmoid_backward)));

}