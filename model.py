import torch
import torch.nn as nn
import torch.utils.data as data
from torch import Tensor
from tqdm import tqdm
from torch.autograd import Variable


def calc_iou(out_mask, true_mask):
    nominator = torch.sum(out_mask * true_mask)
    denominator = torch.sum(out_mask) + torch.sum(true_mask) - nominator
    return nominator/denominator

def put_masks(images: Tensor, masks: Tensor):
    B, C, H, W = images.shape
    masks = masks.reshape((B, 1, H, W))
    return images*masks
    
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
    
    def forward(self, input: Tensor):
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

    def calculate_loss_and_metrics(self, input: Tensor, target: Tensor, loss_fn: torch.nn.Module, *metrics):
        output = self(input)
        loss = loss_fn(output, target)
        result_metrics = []
        for metric in metrics:
            m = metric(output, target)
            result_metrics.append(m)
        return loss, *result_metrics

    def validate(self, valid_loader: data.DataLoader, epoch_idx: int):
        assert self.loss_fn, "Model has no set loss function!"
        valid_loss = 0
        valid_iou = 0
        with torch.no_grad():
            for image, true_mask in tqdm(valid_loader, desc=f"Validating E{epoch_idx+1}"):
                out_mask = self(image)
                loss, iou = self.calculate_loss_and_metrics(image, true_mask, self.loss_fn, calc_iou)
                valid_loss += loss
                valid_iou += iou
        n = len(valid_loader.dataset)
        valid_loss /= n
        valid_iou /= n
        return valid_loss, valid_iou
    
    def _train_epoch(self, train_loader: data.DataLoader, epoch_idx: int, optimizer: torch.optim.Optimizer):
        assert self.loss_fn, "Model has no set loss function!"
        train_loss = 0
        train_iou = 0
        for image, true_mask in tqdm(train_loader, desc=f"Training E{epoch_idx+1}"):
            optimizer.zero_grad()
            image = image.cuda()
            true_mask = true_mask.cuda()
            loss, iou = self.calculate_loss_and_metrics(image, true_mask, self.loss_fn, calc_iou)
            loss.backward()
            optimizer.step()
            train_loss += loss
            train_iou += iou
        n = len(train_loader.dataset)
        train_loss /= n
        train_iou /= n
        return train_loss, train_iou

    def generate_images_from_batch(self, images: Tensor, masks: Tensor=None) -> Tensor:
        if not masks is None:
            return put_masks(images, masks)
        masks = self(images)
        return put_masks(images, masks)

    def train_and_validate(self, train_loader: data.DataLoader, valid_loader: data.DataLoader, epochs: int, optimizer:torch.optim.Optimizer=torch.optim.Adam):
        """Train and validates model, if valid_loader is None then won't perform validation.
        """
        history = {"train_loss": [], "train_iou": [], "valid_loss": [], "valid_iou": []}
        image_history = {"true": [], "predicted": []}
        for epoch_idx in range(epochs):
            # REPORT ONE IMAGE
            self.eval()
            images, true_masks = next(iter(train_loader))
            images, true_masks = images.cuda(), true_masks.cuda()
            estimate_masked = self.generate_images_from_batch(images).detach().cpu()
            del images, true_masks
            # TRAINING
            self.train()
            train_loss, train_iou = self._train_epoch(train_loader, epoch_idx, optimizer)
            # VALIDATE
            # if valid loader not given then we don't perform validation
            if not valid_loader: 
                yield estimate_masked, train_loss, train_iou, .0, .0
                continue
            self.eval()
            valid_loss, valid_iou = self.validate(valid_loader, epoch_idx)
            yield estimate_masked, train_loss, train_iou, valid_loss, valid_iou