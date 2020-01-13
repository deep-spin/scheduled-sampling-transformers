"""
An extended versoin of the Softmax function.
Allows usage of temperature parameter.
"""

import torch
import torch.nn as nn


class SoftmaxWithTemperature(nn.Module):

    def __init__(self, dim=0, alpha=1.0):
        super(SoftmaxWithTemperature, self).__init__()
        self._softmax = nn.Softmax(dim)
        self._alpha = alpha

    def forward(self, input):
        return self._softmax(self._alpha*input)


class GumbelSoftmax(nn.Module):

    def __init__(self, dim=0, alpha=1.0):
        super(GumbelSoftmax, self).__init__()
        self._softmax = nn.Softmax(dim)
        self._alpha = alpha

    def forward(self, input):
        # 1. Sample Gumbel noise with the shape of the given input
        U = torch.rand(input.shape).cuda()
        gumbel_noise = -torch.log(-torch.log(U))
        # 2. Add the Gumbel noise to the scores
        input_with_gumbel_noise = input+gumbel_noise
        # 3. Return softmax of the input with gumbel noise, with the given temperature
        return self._softmax(self._alpha*input_with_gumbel_noise)