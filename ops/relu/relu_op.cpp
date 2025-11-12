#include <torch/extension.h>

torch::Tensor relu_forward(torch::Tensor input);
torch::Tensor relu_backward(torch::Tensor grad_output, torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &relu_forward,  "ReLU forward (CUDA)");
    m.def("backward", &relu_backward, "ReLU backward (CUDA)");
}
