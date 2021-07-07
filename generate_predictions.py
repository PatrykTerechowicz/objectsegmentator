import argparse
import model
import torch
import os
import torch.nn.functional as F
import imageio as iio
import numpy as np
import string
from random import choices
from kornia.morphology import opening
from data_loader import to_tensor, normalize
from torchvision.datasets import ImageFolder
from os.path import exists, join
from custom_activations import HardELU

def code():
    return "".join(choices(string.ascii_letters, k=16))

def get_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("ds_dir", type=str, help="path to dataset")
    parser.add_argument("-load_path", type=str, default="./best.pth", help="path to model weights file")
    parser.add_argument("-out_path", type=str, default="./out/", help="where to save outputs")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    assert exists(args.ds_dir)
    assert exists(args.load_path)
    if not exists(args.out_path):
        try:
            os.makedirs(args.out_path)
        except Exception:
            pass

    ds = ImageFolder(args.ds_dir)
    classes = ds.classes
    for _class in classes:
        class_dir = join(args.out_path, _class)
        if not exists(class_dir): 
            os.makedirs(class_dir)
    net = model.Segmentator(activation_function=HardELU)
    net.load_state_dict(torch.load(args.load_path))
    net.cuda()

    def predict(image: torch.Tensor):
        return net(image)

    def prepare(image):
        image = to_tensor(image)
        C, H, W = image.shape
        small_image = F.interpolate(image.unsqueeze(0), (int(H/4), int(W/4))).squeeze(0)
        return normalize(small_image)

    kernel = torch.ones((7, 7)).cuda()
    for image, target in ds:
        output_mask = predict(prepare(image).cuda())
        output_mask = opening(output_mask, kernel)
        image = to_tensor(image)
        C, H, W = image.shape
        resized_mask = F.interpolate(output_mask, (H, W)).squeeze(0)
        mask = resized_mask > 0.95
        mask = mask.cpu().squeeze(0)
        final_cut = torch.zeros((C+1, H, W))
        final_cut[-1, mask] = 1.0
        final_cut[:3, :, :] = image[:, :, :]
        final_cut = (final_cut*255).permute((1, 2, 0)).numpy().astype(np.uint8)
        final_cut = final_cut[10:-10, 10:-10, :]
        path = join(args.out_path, classes[target], f"{code()}.png")
        iio.imwrite(path, final_cut)