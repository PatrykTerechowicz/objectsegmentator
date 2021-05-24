import torch
from torch import tensor
import torch.nn as nn
import torch.utils.data as data
import tqdm
import gc
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

lossl1 = torch.nn.BCELoss(reduce='none')

def dice_loss(out_mask, true_mask):
    numerator = 2 * torch.sum(out_mask * true_mask)
    denominator = torch.sum(out_mask + true_mask)
    return 1 - numerator/denominator

def calc_iou(out_mask, true_mask):
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

    def train(self, train_loader: data.DataLoader, valid_loader: data.DataLoader, epochs: int=10, lr: float=1e-2, loss_fn=lossl1, summary: SummaryWriter=None, optim:torch.optim.Optimizer=torch.optim.Adam):
        """Train and validates model, if valid_loader is None then won't perform validation.
        """
        optimizer = optim(self.parameters(), lr=lr, momentum=0.9)
        s = f"optimizer: {type(optimizer)}\nloss: {type(loss_fn)}\n"
        summary.add_text("desc", s)
        for epoch_idx in range(epochs):
            # report one image
            for data in train_loader:
                image, true_mask = data
                est_mask = self(image)
                images_true = image*(true_mask.unsqueeze(1))
                images_est = image*(est_mask.unsqueeze(1))
                grid_true = make_grid(images_true, nrow=9)
                summary.add_image("true_objects", grid_true, global_step=epoch_idx)
                grid_est = make_grid(images_est, nrow=9)
                summary.add_image("estimated_objects", grid_est, global_step=epoch_idx)
                break
            train_loss = 0
            train_iou = 0
            for image, true_mask in tqdm.tqdm(train_loader):
                optimizer.zero_grad()
                image = image.cuda()
                true_mask = true_mask.cuda()
                out_mask = self(image)
                loss = torch.mean(loss_fn(out_mask, true_mask))
                loss.backward()
                optimizer.step()
                iou = calc_iou(out_mask, true_mask)
                train_loss += loss
                train_iou += iou
            n = len(train_loader.dataset)
            train_loss /= n
            train_iou /= n
            if summary:
                summary.add_scalars("train", {"loss": train_loss, "iou": train_iou}, epoch_idx)
            if not valid_loader: continue
            valid_loss = 0
            valid_dice = 0
            valid_iou = 0
            with torch.no_grad():
                for image, true_mask in tqdm.tqdm(valid_loader):
                    out_mask = self(image)
                    loss = torch.mean(loss_fn(out_mask, true_mask))
                    dice = dice_loss(out_mask, true_mask)
                    iou = calc_iou(out_mask, true_mask)
                    valid_loss += loss
                    valid_dice += dice
                    valid_iou += iou
            n = len(valid_loader.dataset)
            valid_loss /= n
            valid_dice /= n
            valid_iou /= n
            if summary:
                summary.add_scalars("valid", {"loss": valid_loss, "iou": valid_iou, "dice": valid_dice}, epoch_idx)