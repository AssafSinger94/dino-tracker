import torch
from torch import nn
import torch.nn.functional as F
import math


class NormalizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(NormalizedConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) # C_out x C_in x K x K
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels)) # C_out
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def get_weight_sum(self):
        EPS = 1e-8
        w_sum = self.weight.sum(dim=[2,3])[:, :, None, None]
        unstable_indices = w_sum.abs()<EPS
        if unstable_indices.sum() > 0:
            w_sum[unstable_indices] = torch.sign(w_sum[unstable_indices]) * EPS
        return w_sum

    def forward(self, x):
        w_sum = self.get_weight_sum()
        normalized_weights = self.weight / w_sum
        
        return F.conv2d(x, normalized_weights, bias=self.bias, stride=self.stride, padding=self.padding)
