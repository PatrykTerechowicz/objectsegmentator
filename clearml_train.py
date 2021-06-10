from json import load
from clearml import Task, Logger
from argparse import ArgumentParser
from data_loader import ObjectSegmentationDataset
from typing import Optional
from os.path import join
from torch.utils.data import DataLoader
from model import Segmentator
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from plotting import create_grid
from datetime import datetime
import torch.nn as nn
import torch
import data_loader
import losses

CUDA = torch.cuda.is_available()

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("out_path", type=str)
    parser.add_argument("train_ds", type=str)
    parser.add_argument("-valid_ds", default=None, type=str)
    parser.add_argument("-lr", default=0.001, type=float)
    parser.add_argument("-loss_fn", default="bce_dice", type=str)
    parser.add_argument("-epochs", default=10, type=int)
    parser.add_argument("-batch_size", default=9, type=int)
    parser.add_argument("-load_memory", action="store_true")
    args = parser.parse_args()
    return args

def init_task():
    task: Task = Task.init("Foreground Segmentation", "Train")
    return task, task.get_logger()

def load_datasets(train_ds_path: str, train_loader_params: dict, load_memory: bool=False, valid_ds_path: Optional[str] = None, valid_loader_params: dict={}):
    train_ds = ObjectSegmentationDataset(train_ds_path, join(train_ds_path, "annotations.json"), load_memory=load_memory)
    train_loader = DataLoader(train_ds, **train_loader_params)
    valid_loader = None
    if valid_ds_path:
        valid_ds = ObjectSegmentationDataset(valid_ds_path, join(valid_ds_path, "annotations.json"), load_memory=load_memory)
        valid_loader = DataLoader(valid_ds, **valid_loader_params)
    return train_loader, valid_loader


def main():
    args = get_arguments()
    task, logger = init_task()
    loader_params = {
        "batch_size": args["batch_size"],
        "num_workers": 4,
        "pin_memory": torch.cuda.is_available()
    }
    optimizer = Adam
    activation_function = nn.ReLU6
    augment = data_loader.weak_augment
    loss_fn = losses.ComboLoss()
    hyper_params = {
        "loss_fn": type(loss).__name__,
        "optimizer": optimizer.__name__,
        "activation_function": activation_function.__name__,
        "augment strategy": augment.__name
        }
    task.connect(hyper_params, hyper_params)
    train_loader, valid_loader = load_datasets(args.train_ds, loader_params, load_memory=args.load_memory, valid_ds_path=args.valid_ds, valid_loader_params=loader_params)
    model: Segmentator = Segmentator(activation_function=activation_function)
    model.loss_fn = loss_fn
    optimizer = optimizer(model.parameters(), lr=args.lr)
    for n, metrics in enumerate(model.train_and_validate(train_loader, valid_loader, optimizer, args.epochs, denormalizer=data_loader.denormalize, transform_data=augment)):
        estimate_masked, train_loss, train_iou, valid_loss, valid_iou = metrics
        logger.add_scalar("loss", "train", train_loss, iteration=n)
        logger.add_scalar("loss", "valid", valid_loss, iteration=n)
        logger.add_scalar("iou", "train", train_iou, iteration=n)
        logger.add_scalar("iou", "valid", valid_iou, iteration=n)
        grid = create_grid(estimate_masked)
        logger.report_image("estimates", "train", iteration=n, matrix=grid.permute((1, 2, 0)).cpu().numpy())
    date_s: str = datetime.now().strftime("%d-%m-%y_%H-%M")
    out_dir = join(args.out_path, "fg_segmentator", date_s, "best.pth")
    torch.save(model.state_dict(), out_dir)

if __name__ == "__main__":
    main()