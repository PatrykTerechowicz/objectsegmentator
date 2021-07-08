import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

def dice_loss(scale=None):
    def fn(input, target):
        smooth = 1.

        if scale is not None:
            scaled = F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=False)
            iflat = scaled.view(-1)
        else:
            iflat = input.view(-1)

        iflat = iflat.float()
        tflat = target.view(-1).float()
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

    return fn