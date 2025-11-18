import torch

class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(context, x):
        context.save_for_backward(x)
        return torch.ops.pytorch_cuda_ops.sigmoid(x)
    
    @staticmethod
    def backward(context, grad_output):
        (x, ) = context.saved_tensors
        return torch.ops.pytorch_cuda_ops.sigmoid_backward(grad_output.contiguous(), x)
    
def sigmoid(x):
    return SigmoidFunction.apply(x)