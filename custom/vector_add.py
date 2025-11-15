import torch

class VectorAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(context, a, b):
        out = torch.ops.pytorch_cuda_ops.vector_add(a, b)
        return out
    
    @staticmethod
    def backward(context, grad_out):
        grad_a = grad_out
        grad_b = grad_out
        return grad_a, grad_b
    
# wrapper
def vector_add(a, b): 
    return VectorAddFunction.apply(a, b)
