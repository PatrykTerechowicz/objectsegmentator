import torch
import torch.nn as nn
import json
import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
from torch.utils import data
from typing import List, Tuple



to_tensor = ToTensor()

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
    region_tensor[mask==0] = -1


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
    def __init__(self, ds_dir: str, annotation_path: str, load_memory=False, preprocess = None, normalize=None, augment = None):
        super(ObjectSegmentationDataset, self).__init__()
        self.ds_dir = ds_dir
        if not os.path.exists(annotation_path):
            raise FileExistsError("Cant find annotation file!")
        self.annotations = json.load(open(annotation_path))
        self.data_raw: List[Tuple[str, List]] = []
        img_metadata = self.annotations["_via_img_metadata"]
        for key in img_metadata:
            image = img_metadata[key]
            filename = image["filename"]
            regions = image["regions"]
            self.data_raw.append((filename, regions))
        self.load_memory = load_memory
        if self.load_memory:
            self.data = []
            for data in tqdm(self.data_raw):
                filename, regions = data
                image = load_image(os.path.join(ds_dir, filename))
                mask = create_mask(image, regions)
                if preprocess:
                    image = preprocess(image)
                    mask = preprocess(mask.unsqueeze(1)).squeeze(1)
                self.data.append((image, mask))

    def __len__(self):
        if self.load_memory: return len(self.data)
        return len(self.data_raw)
    
    def __getitem__(self, index):
        if self.load_memory: 
            image, mask = self.data[index]
            return image, mask
        filename, regions = self.data_raw[index]
        image = load_image(os.path.join(self.ds_dir, filename))
        mask = create_mask(image, regions)
        return image, mask

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