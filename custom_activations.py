import torch
import torch.nn as nn

class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, input):
        return torch.relu(input)-torch.exp(torch.neg(torch.square(input)))