import torch
from torch import tensor
import torch.nn as nn
import torch.utils.data as data
import tqdm
from torch.utils.tensorboard import SummaryWriter
from sam import SAM
from torch.autograd import Variable

def dice_loss(out_mask, true_mask):
    numerator = 2 * torch.sum(out_mask * true_mask)
    denominator = torch.sum(out_mask + true_mask)
    return 1 - numerator/denominator

def iou(out_mask, true_mask):
    predicted1 = out_mask > 0.5
    true1 = true_mask > 0.99
    nominator = torch.sum(true1 == predicted1)
    denominator = torch.sum(predicted1) + torch.sum(true1) - nominator
    return nominator/denominator

class Segmentator(nn.Module):
    def __init__(self) -> None:
        super(Segmentator, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(11, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(19, 8, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(27, 8, 3, stride=1, padding=1)
        self.finconv1 = nn.Conv2d(35, 16, 5, stride=1, padding=2)
        self.finconv2 = nn.Conv2d(16, 1, 5, stride=1, padding=2)
        self.vec = Variable(torch.randn(1, 16, 1, 1), requires_grad=True)
        self.act = nn.Hardswish()
        self.out_act = nn.Sigmoid()
        self.debug = False
    
    def toggle_debug(self):
        self.debug = not self.debug
        return self.debug
    
    def forward(self, input: torch.Tensor):
        B, C, H, W = input.shape
        f1 = self.conv1(input)
        f1 = self.act(f1)
        x = torch.cat([input, f1], dim=1)
        f2 = self.conv2(x)
        f2 = self.act(f2)
        x = torch.cat([x, f2], dim=1)
        f3 = self.conv3(x)
        f3 = self.act(f3)
        x = torch.cat([x, f3], dim=1)
        f4 = self.conv4(x)
        f4 = self.act(f4)
        x = torch.cat([x, f4], dim=1)
        x = self.finconv1(x)
        x = self.act(x)
        output = self.finconv2(x)[:, 0, ...]
        output = self.out_act(output)
        if self.debug:
            return f1, f2, f3, f4, output
        return output

    def train(self, train_loader: data.DataLoader, valid_loader: data.DataLoader, epochs: int=10, lr: float=1e-4, summary: SummaryWriter=None):
        """Train and validates model, if valid_loader is None then won't perform validation.
        """
        base_optimizer = torch.optim.SGD()
        optimizer = SAM(self.parameters(), base_optimizer, lr=lr, momentum=0.9)
        history = {"train_dice": [], "valid_dice": [], "train_iou": [], "valid_iou": []}
        for epoch_idx in range(epochs):
            for batch_idx, batch in tqdm.tqdm(train_loader):
                image, true_mask = batch
                out_mask = self(image)
                loss = dice_loss(out_mask, true_mask)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                dice_loss(out_mask, true_mask).backward()
                optimizer.second_step(zero_grad=True)
                history["train_dice"].append(loss)
                iou_m = iou(out_mask, true_mask)
                history["train_iou"].append(iou_m)
                if summary: summary.add_scalars("train", {"dice": loss, "iou": iou_m})
            if not valid_loader: continue
            with torch.no_grad():
                for batch_idx, batch in tqdm.tqdm(valid_loader):
                    image, true_mask = batch
                    out_mask = self(image)
                    loss = dice_loss(out_mask, true_mask)
                    history["valid_dice"].append(loss)
                    iou_m = iou(out_mask, true_mask)
                    history["valid_iou"].append(iou_m)
                    if summary: summary.add_scalars("valid", {"dice": loss, "iou": iou_m})