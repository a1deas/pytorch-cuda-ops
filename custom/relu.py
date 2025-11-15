import torch

class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(context, x):
        context.save_for_backward(x)
        return torch.ops.pytorch_cuda_ops.relu(x)
    
    @staticmethod
    def backward(context, grad_out):
        (x, ) = context.saved_tensors
        return torch.ops.pytorch_cuda_ops.relu_backward(grad_out.contiguous(), x)
    
def relu(x):
    return ReLUFunction.apply(x)