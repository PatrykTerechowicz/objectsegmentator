from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import numpy as np
from random import random
from tqdm import tqdm
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
from torch.utils import data
from typing import List, Tuple, Optional
from kornia.augmentation.augmentation import Denormalize, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, GaussianBlur, Normalize, RandomResizedCrop, RandomRotation, RandomElasticTransform

normalize = Normalize(torch.from_numpy(np.array([0.485, 0.456, 0.406])), torch.from_numpy(np.array([0.229, 0.224, 0.225])))
denormalize = Denormalize(torch.from_numpy(np.array([0.485, 0.456, 0.406])), torch.from_numpy(np.array([0.229, 0.224, 0.225])))

to_tensor = ToTensor()
clr_jitter = ColorJitter(0.1, 0.1, 0.1, 0.15, p=0.77)
blur = GaussianBlur(kernel_size=(3,9), sigma=(.5, 1), p=0.328)
crop1 = RandomResizedCrop((224, 224), scale=(0.35, 0.7))
crop2 = RandomResizedCrop((320, 320), scale=(0.35, 0.7))
rot = RandomRotation(degrees=(-5, 5), p=.14)
def weak_augment(image_batch, mask_batch):
    image_batch = clr_jitter(image_batch)
    image_batch = blur(image_batch)
    image_batch = normalize(image_batch)
    if random()>0.5:
        image_batch = crop1(image_batch)
        mask_batch = crop1(mask_batch, crop1._params)
    else:
        image_batch = crop2(image_batch)
        mask_batch = crop2(mask_batch, crop2._params)
    return image_batch, mask_batch

elastic = RandomElasticTransform(p=0.32, sigma=(32, 32))
rh_flip = RandomHorizontalFlip(p=0.5)
rv_flip = RandomVerticalFlip(p=0.5)
def strong_augment(image_batch, mask_batch):
    image_batch = clr_jitter(image_batch)
    image_batch = blur(image_batch)
    image_batch = elastic(image_batch)
    mask_batch = elastic(mask_batch, elastic._params)
    image_batch = normalize(image_batch)
    image_batch = rh_flip(image_batch)
    mask_batch = rh_flip(mask_batch, rh_flip._params)
    image_batch = rv_flip(image_batch)
    mask_batch = rv_flip(mask_batch, rv_flip._params)
    if random()>0.5:
        image_batch = crop1(image_batch)
        mask_batch = crop1(mask_batch, crop1._params)
    else:
        image_batch = crop2(image_batch)
        mask_batch = crop2(mask_batch, crop2._params)
    return image_batch, mask_batch


def load_image(file_path):
    pilim = Image.open(file_path).convert("RGB")
    tensorim = to_tensor(pilim)
    return tensorim

def polygon_from_allpoints(shape_attributes):
    x_points = shape_attributes["all_points_x"]
    y_points = shape_attributes["all_points_y"]
    points = []
    for x, y in zip(x_points, y_points):
        points.append((x, y))
    return points

def put_region(r, region_tensor):
    polygon = polygon_from_allpoints(r["shape_attributes"])
    H, W = region_tensor.shape
    mask_img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
    mask = torch.from_numpy(np.array(mask_img))
    region_tensor[mask==1] = 1


def create_mask(image, regions):
    C, H, W = image.shape
    mask = torch.zeros((H, W))
    for r in regions:
        put_region(r, mask)
    return mask


class ObjectSegmentationDataset(data.Dataset):
    """It will load everything into memory if load_memory=True.
    Args:
        data ([type]): [description]
    """
    def __init__(self, ds_dir: str, annotation_path: Optional[str]=None, load_memory=False):
        super(ObjectSegmentationDataset, self).__init__()
        self.ds_dir = ds_dir
        if not annotation_path:
            annotation_path = os.path.join(ds_dir, "annotations.json")
        if not os.path.exists(annotation_path):
            raise FileExistsError("Cant find annotation file!")
        self.annotations = json.load(open(annotation_path))
        self.data_raw: List[Tuple[str, List]] = []
        img_metadata = self.annotations["_via_img_metadata"]
        for key in img_metadata:
            image = img_metadata[key]
            filename = image["filename"]
            regions = image["regions"]
            if os.path.exists(os.path.join(ds_dir, filename)): self.data_raw.append((filename, regions))
        self.load_memory = load_memory
        if self.load_memory:
            self.data = []
            for data in tqdm(self.data_raw):
                filename, regions = data
                image = load_image(os.path.join(ds_dir, filename))
                mask = create_mask(image, regions)
                self.data.append((image, mask))

    def __len__(self):
        if self.load_memory: return len(self.data)
        return len(self.data_raw)
    
    def __getitem__(self, index):
        if self.load_memory: 
            image, mask = self.data[index]
        else:
            filename, regions = self.data_raw[index]
            image = load_image(os.path.join(self.ds_dir, filename))
            mask = create_mask(image, regions)
        return image, mask.unsqueeze_(0)

if __name__ == "__main__":
    import time
    T1 = time.time()
    ds = ObjectSegmentationDataset(r"E:\Datasets\100Objects_train", r"E:\Datasets\100Objects_train\annotations.json", load_memory=True)
    dl = data.DataLoader(ds, batch_size=2, num_workers=0)
    T2 = time.time()
    elapsed_time = T2-T1
    print(f"Elapsed Time {elapsed_time:.2f}")
    for i, m in tqdm(dl):
        print(i.device)