import torch
from torch import Tensor
import torch.nn.functional as F

def bce_dice(outputs, targets, eps=1e-15, smooth=1., w=1/3):
        targets = targets.view(-1)
        outputs = outputs.view(-1)
        bce = F.binary_cross_entropy(outputs, targets, reduction="mean")
        intersection = (outputs * targets).sum()
        union = outputs.sum() + targets.sum()
        dice = (2. * intersection + smooth)/(union + eps + smooth)
        return w * bce - (1 - w)*torch.log(dice)


loss = {"bce_dice": bce_dice}