import torch
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import torch.nn as nn

class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.elu = nn.ELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.elu(out)
        return out

class DQN_ICM(torch.nn.Module):
    def __init__(self, input_shape, layers, kernel_sizes, strides, fc_dim, out_dim, padding, device):
        super(DQN_ICM, self).__init__()

        self.feature_extractor = torch.nn.Sequential(
                ConvBlock(in_planes=input_shape, out_channels=layers[0], kernel_size=kernel_sizes[0], stride=strides[0], padding=padding[0]),
                ConvBlock(in_planes=layers[0], out_channels=layers[1], kernel_size=kernel_sizes[1], stride=strides[1], padding=padding[0]),
                ConvBlock(in_planes=layers[1], out_channels=layers[2], kernel_size=kernel_sizes[2], stride=strides[2], padding=padding[0]),
                ConvBlock(in_planes=layers[2], out_channels=layers[3], kernel_size=kernel_sizes[3], stride=strides[3], padding=padding[0]),
                torch.nn.Flatten(),
                )

        self.final = torch.nn.Sequential(
                nn.Linear(288, fc_dim),
                nn.ELU(),
                nn.Linear(fc_dim, out_dim)
            )


    def forward(self,x):
        out = self.feature_extractor(x)
        out = self.final(out)
        #The output is of shape N x 12.
        return out
    
