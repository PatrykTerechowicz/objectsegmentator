import torch
from torch import tensor
import torch.nn as nn
import torch.utils.data as data
import tqdm
import gc
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from sam import SAM
from torch.autograd import Variable

lossl1 = torch.nn.L1Loss()

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

def put_masks(images, masks):
    B, C, H, W = images.shape
    for b in range(B):
        mask = masks[b] > 0.5
        images[b, :, mask] = 0
    return images
    
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
        x = torch.cat([input, self.act(self.conv1(input))], dim=1)
        x = torch.cat([x, self.act(self.conv2(x))], dim=1)
        x = torch.cat([x, self.act(self.conv3(x))], dim=1)
        x = torch.cat([x, self.act(self.conv4(x))], dim=1)
        x = self.finconv1(x)
        x = self.act(x)
        output = self.finconv2(x)[:, 0, ...]
        output = self.out_act(output)
        return output

    def train(self, train_loader: data.DataLoader, valid_loader: data.DataLoader, epochs: int=10, lr: float=1e-2, loss_fn=lossl1, summary: SummaryWriter=None):
        """Train and validates model, if valid_loader is None then won't perform validation.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, momentum=0.9)
        history = {"train_dice": [], "valid_dice": [], "train_iou": [], "valid_iou": [], "valid_loss": []}
        for epoch_idx in range(epochs):
            i = 0
            for image, true_mask in tqdm.tqdm(train_loader):
                i+=1
                optimizer.zero_grad()
                image = image.cuda()
                true_mask = true_mask.cuda()
                out_mask = self(image)
                loss = loss_fn(out_mask, true_mask)
                loss.backward()
                optimizer.step()
                history["train_loss"].append(loss)
                iou_m = iou(out_mask, true_mask)
                history["train_iou"].append(iou_m)
                if summary: 
                    summary.add_scalars("train", {"dice": loss, "iou": iou_m})
            if not valid_loader: continue
            with torch.no_grad():
                for image, true_mask in tqdm.tqdm(valid_loader):
                    out_mask = self(image)
                    loss = loss_fn(out_mask, true_mask)
                    history["valid_loss"].append(loss)
                    dice = dice_loss(out_mask, true_mask)
                    history["valid_dice"].append(dice)
                    iou_m = iou(out_mask, true_mask)
                    history["valid_iou"].append(iou_m)
                    if summary: summary.add_scalars("valid", {"dice": dice, "iou": iou_m, "loss": loss})