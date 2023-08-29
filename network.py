import numpy as np
import math

import torch
import torch.nn as nn


########################################################################

def gen_mask(row, col, percent=0.5, num_zeros=None):
    if num_zeros is None:
        # Total number being masked is 0.5 by default.
        num_zeros = int((row * col) * percent)

    mask = np.hstack([np.zeros(num_zeros),
                      np.ones(row * col - num_zeros)])
    np.random.shuffle(mask)
    return mask.reshape(row, col)

def coth(x):
    return np.cosh(x)/np.sinh(x)

########################################################################


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            weight = weight * mask
        
        output = input.mm(weight.t().float())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        ctx.save_for_backward(input, weight, bias, mask)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight.float())
        
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                grad_weight = grad_weight * mask
        
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
            
        return grad_input, grad_weight, grad_bias, grad_mask
    
class CustomizedLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True, mask=None):
        super(CustomizedLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.weight = nn.Parameter(torch.Tensor(
            self.output_features, self.input_features
        ))
        
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.output_features)
            )
        else:
            self.register_parameter('bias',None)
            
        self.init_params()
        
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.float).t()
            self.mask = nn.Parameter(mask, requires_grad=False)
        else:
            self.register_parameter('mask', None)
            
    def init_params(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input):
        return LinearFunction.apply(
            input, self.weight, self.bias, self.mask
        )
    
    def extra_repr(self):
        return 'input_features={}, output_features={}, bias={}, mask={}'.format(
            self.input_features, self.output_features,
            self.bias is not None, self.mask is not None
        )

class Network(nn.Module):
    def __init__(self, in_size, out_size, ratio=[0, 0, 0, 0]):
        super(Network, self).__init__()
        self.fc1 = CustomizedLinear(in_size, 50, 
                                    mask=gen_mask(in_size, 50, ratio[0]))
        self.bn1 = nn.BatchNorm1d(50)
        
        self.fc2 = CustomizedLinear(50, 50, 
                                    mask=gen_mask(50, 50, ratio[1]))
        self.bn2 = nn.BatchNorm1d(50)
        
        self.fc3 = CustomizedLinear(50, 50, 
                                    mask=gen_mask(50, 50, ratio[2]))
        self.bn3 = nn.BatchNorm1d(50)
        
        self.fc4 = CustomizedLinear(50, out_size, 
                                    mask=gen_mask(50, out_size, ratio[3]))
        self.bn4 = nn.BatchNorm1d(out_size)
        
        self.relu = nn.ReLU()

        # Initialize parameters by following steps:
        # Link: https://pytorch.org/docs/stable/nn.init.html
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=0, b=1)
                # nn.init.kaiming_normal_(m.weight,
                #                         mode='fan_out',
                #                         nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        return x