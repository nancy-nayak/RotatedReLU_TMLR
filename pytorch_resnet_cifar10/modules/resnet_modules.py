import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable
from scipy.stats import ortho_group

# class probrelu(nn.Module):
#     def __init__(self):
#         super(probrelu, self).__init__()

    
#     def forward(self, input):
#         output = Probrelu_Activation().apply(input)
#         return output

# class Probrelu_Activation(Function):
#     @staticmethod
#     def forward(ctx, input):
#         out = 
#         return out


class hybridreluwflat(nn.Module):
    def __init__(self, a, b, c, d):
        super(hybridreluwflat, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def forward(self, input):
        output = Hybrid_Relu_WFlat_Activation().apply(input, self.a, self.b, self.c, self.d)
        return output



## activation with hybrid relu
class Hybrid_Relu_WFlat_Activation(Function):
    @staticmethod
    def forward(ctx, input, a, b, c, d):
        ctx.save_for_backward(input, a, b, c, d)
        out = torch.relu(a*input + c) -torch.relu(-b*input - c*b/a - d)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, a, b, c, d = ctx.saved_tensors
    
        grad_input = grad_output.clone()
        grad_input[torch.logical_and(c+a*input>0.0, b*c/a+d+b*input<0.0)] *= a+b
        grad_input[torch.logical_and(c+a*input>0.0,  b*c/a+d+b*input>=0.0)] *= a
        grad_input[torch.logical_and(c+a*input<=0.0, b*c/a+d+b*input<0.0)] *= b
        grad_input[torch.logical_and(c+a*input<=0.0,  b*c/a+d+b*input>=0.0)] = 0.0

        grad_a = grad_output.clone()
        grad_a[torch.logical_and(c+a*input>0.0, b*c/a+d+b*input<0.0)]*= input[torch.logical_and(c+a*input>0.0, b*c/a+d+b*input<0.0)]-b*c/a**2
        grad_a[torch.logical_and(c+a*input>0.0, b*c/a+d+b*input>=0.0)]*= input[torch.logical_and(c+a*input>0.0, b*c/a+d+b*input>=0.0)]
        grad_a[torch.logical_and(c+a*input<=0.0, b*c/a+d+b*input<0.0)]*= -b*c/a**2
        grad_a[torch.logical_and(c+a*input<=0.0, b*c/a+d+b*input>=0.0)] = 0.0
       
        grad_b = grad_output.clone()
        grad_b[b*c/a+d+b*input<0.0]*= input[b*c/a+d+b*input<0.0] + c/a
        grad_b[b*c/a+d+b*input>=0.0] = 0.0

        grad_c = grad_output.clone()
        grad_c[torch.logical_and(c+a*input>0.0, b*c/a+d+b*input<0.0)]*= 1.0+b/a
        grad_c[torch.logical_and(c+a*input>0.0, b*c/a+d+b*input>=0.0)]*= 1.0
        grad_c[torch.logical_and(c+a*input<=0.0, b*c/a+d+b*input<0.0)]*= b/a
        grad_c[torch.logical_and(c+a*input<=0.0, b*c/a+d+b*input>=0.0)] = 0.0

        grad_d = grad_output.clone()
        grad_d[b*c/a+d+b*input<0.0]*= 1.0
        grad_d[b*c/a+d+b*input>=0.0]= 0.0
        return grad_input, grad_a, grad_b, grad_c, grad_d




    