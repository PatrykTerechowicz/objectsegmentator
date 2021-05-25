import torch
from torch import tensor
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import gc
from torchvision.utils import make_grid
from torch.autograd import Variable

lossl1 = torch.nn.BCELoss(reduce='none')

def calc_iou(out_mask, true_mask):
    nominator = torch.sum(out_mask * true_mask)
    denominator = torch.sum(out_mask) + torch.sum(true_mask) - nominator
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

    def calculate_loss_and_metrics(self, input: torch.Tensor, target: torch.Tensor, loss_fn: torch.nn.Module, *metrics):
        output = self(input)
        loss = loss_fn(output, target)
        result_metrics = []
        for metric in metrics:
            m = metric(input, target)
            result_metrics.append(m)
        return loss, *result_metrics

    def train(self, train_loader: data.DataLoader, valid_loader: data.DataLoader, epochs: int=10, lr: float=1e-2, loss_fn: torch.nn.Module=lossl1, optim:torch.optim.Optimizer=torch.optim.Adam):
        """Train and validates model, if valid_loader is None then won't perform validation.
        """
        optimizer = optim(self.parameters(), lr=lr)
        history = {"train_loss": [], "train_iou": [], "valid_loss": [], "valid_iou": []}
        image_history = {"true": [], "predicted": []}
        for epoch_idx in range(epochs):
            # report one image
            for data in train_loader:
                image, true_mask = data
                image = image.cuda()
                true_mask = true_mask.cuda()
                est_mask = self(image)
                images_true = image*(true_mask.unsqueeze(1))
                images_est = image*(est_mask.unsqueeze(1))
                grid_true = make_grid(images_true, nrow=3)
                grid_est = make_grid(images_est, nrow=3)
                image_history["true"].append(grid_true)
                image_history["predicted"].append(grid_est)
                break
            train_loss = 0
            train_iou = 0
            for image, true_mask in tqdm(train_loader, desc=f"Training E{epoch_idx+1}"):
                optimizer.zero_grad()
                image = image.cuda()
                true_mask = true_mask.cuda()
                loss, iou = self.calculate_loss_and_metrics(image, true_mask, loss_fn, calc_iou)
                loss.backward()
                optimizer.step()
                train_loss += loss
                train_iou += iou
            n = len(train_loader.dataset)
            train_loss /= n
            train_iou /= n
            history["train_loss"].append(train_loss)
            history["train_iou"].append(train_iou)
            if not valid_loader: continue
            valid_loss = 0
            valid_iou = 0
            with torch.no_grad():
                for image, true_mask in tqdm(valid_loader, desc=f"Validating E{epoch_idx+1}"):
                    out_mask = self(image)
                    loss, iou = self.calculate_loss_and_metrics(image, true_mask, loss_fn, calc_iou)
                    valid_loss += loss
                    valid_iou += iou
            n = len(valid_loader.dataset)
            valid_loss /= n
            valid_iou /= n
            history["valid_loss"].append(valid_loss)
            history["valid_iou"].append(valid_iou)
        return history, image_history