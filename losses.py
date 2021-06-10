import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

def dice_loss(outputs, targets, smooth=1.0, eps=1e-15):
        intersection = (outputs*targets).sum()
        union = outputs.sum() + targets.sum()
        return 1-(2*intersection+smooth)/(union+eps+smooth)

class ComboLoss(nn.Module):
    def __init__(self):
        super(ComboLoss, self).__init__()
        self.w = 1/10

    def forward(self, outputs, targets):
        targets = targets.view(-1)
        outputs = outputs.view(-1)
        dice = dice_loss(outputs, targets)
        bce = F.binary_cross_entropy(outputs, targets, reduction="mean")
        return self.w*bce+(1-self.w)*dice