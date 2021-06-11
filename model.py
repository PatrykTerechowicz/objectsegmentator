from numpy import floor
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn import Conv2d
from torch import Tensor
from tqdm import tqdm
from torch.autograd import Variable
from typing import Optional, Callable, List, Tuple
import torch.nn.functional as F


def calc_params(model):
    s = 0
    for p in model.parameters():
        s += p.numel()
    return s

def calc_iou(out_mask, true_mask):
    intersection = out_mask * true_mask
    union = out_mask + true_mask - intersection
    return intersection/union

def put_masks(images: Tensor, masks: Tensor):
    B, C, H, W = images.shape
    masks = masks.reshape((B, 1, H, W))
    return images*masks

class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )

class TransposeConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 2,
        stride: int = 2,
        groups: int = 1,
        padding: int = 0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(TransposeConvBNActivation, self).__init__(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding=padding, dilation=dilation, groups=groups),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )

class TransposedInvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, expand_ratio, norm_layer: Optional[Callable[..., nn.Module]]=None):
        super(TransposedInvertedResidual, self).__init__()
        layers: List[nn.Module] = []
        hidden_dim = int(round(in_planes*expand_ratio))
        if expand_ratio != 1:
            layers.append(ConvBNActivation(in_planes, hidden_dim, kernel_size=1))
        layers.extend([
            TransposeConvBNActivation(hidden_dim, hidden_dim, kernel_size=2, stride=2, groups=hidden_dim),
            Conv2d(hidden_dim, out_planes, kernel_size=1)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expand_ratio, norm_layer: Optional[Callable[..., nn.Module]]=None, activation_function: Optional[Callable[..., nn.Module]]=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.res_connect = True if stride == 1 and out_planes == in_planes else False
        hidden_dim = int(round(in_planes*expand_ratio))
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNActivation(in_planes, hidden_dim, kernel_size=1, activation_layer=activation_function))
        layers.extend([ConvBNActivation(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, activation_layer=activation_function),
            Conv2d(hidden_dim, out_planes, kernel_size=1)
            ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        if self.res_connect: return x + self.conv(x)
        return self.conv(x)


class Segmentator(nn.Module):
    def __init__(self, in_channels=3, layers_settings: List[Tuple] = None, activation_function: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Segmentator, self).__init__()
        if not layers_settings:
            layers_settings = [ # t c n s
                    (1, 16, 1, 1),
                    (6, 24, 2, 2),
                    (6, 32, 3, 1),
                    (6, 64, 4, 1),
                    (6, 120, 1, 1),
                    (6, 320, 1, 1),
                    (6, 1, 1, 1)
                ]
        layers = []
        for t, c, n, s in layers_settings:
            for i in range(n):
                layer = InvertedResidual(in_channels, c, s, t, activation_function=activation_function)
                in_channels = c
                layers.append(layer)
        self.layers = nn.Sequential(*layers)
        self.sig = nn.Sigmoid()
    
    def forward(self, input: Tensor):
        small_mask = self.sig(self.layers(input))
        B, C, H, W = input.shape
        big_mask = F.interpolate(small_mask, size=(H, W))
        return big_mask


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
        return valid_loss.cpu().item(), valid_iou.cpu().item()
    
    def _train_epoch(self, train_loader: data.DataLoader, epoch_idx: int, optimizer: torch.optim.Optimizer, transform_data):
        assert self.loss_fn, "Model has no set loss function!"
        train_loss = 0
        train_iou = 0
        for image, true_mask in tqdm(train_loader, desc=f"Training E{epoch_idx+1}"):
            optimizer.zero_grad()
            if transform_data:
                image, true_mask = transform_data(image, true_mask)
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
        return train_loss.cpu().item(), train_iou.cpu().item()

    def generate_images_from_batch(self, images: Tensor, masks: Tensor=None, denormalizer=None) -> Tensor:
        if not masks is None:
            return put_masks(images, masks)
        masks = self(images)
        if denormalizer: images = denormalizer(images)
        return put_masks(images, masks)

    def train_and_validate(self, train_loader: data.DataLoader, valid_loader: data.DataLoader, optimizer:torch.optim.Optimizer, epochs: int, denormalizer=None, transform_data=None):
        """
            Train and validates model, if valid_loader is None then won't perform validation.
        """
        history = {"train_loss": [], "train_iou": [], "valid_loss": [], "valid_iou": []}
        image_history = {"true": [], "predicted": []}
        for epoch_idx in range(epochs):
            # REPORT ONE IMAGE
            self.eval()
            images, true_masks = next(iter(train_loader))
            images, true_masks = transform_data(images, true_masks)
            images, true_masks = images.cuda(), true_masks.cuda()
            estimate_masked = self.generate_images_from_batch(images, denormalizer=denormalizer).detach().cpu()
            del images, true_masks
            # TRAINING
            self.train()
            train_loss, train_iou = self._train_epoch(train_loader, epoch_idx, optimizer, transform_data=transform_data)
            # VALIDATE
            # if valid loader not given then we don't perform validation
            if not valid_loader: 
                yield estimate_masked, train_loss, train_iou, .0, .0
                continue
            self.eval()
            valid_loss, valid_iou = self.validate(valid_loader, epoch_idx)
            yield estimate_masked, train_loss, train_iou, valid_loss, valid_iou