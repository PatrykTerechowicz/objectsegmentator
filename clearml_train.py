from clearml import Task, Logger
# STDLIB
import os
from argparse import ArgumentParser
from os.path import join, exists
from typing import Optional
from datetime import datetime
# CUSTOM
from models import net, losses
from data.data_loader import ObjectSegmentationDataset, strong_augment, just_preprocess
from utils.trainer import Trainer
import utils.metrics as metrics
import utils.const as const
# TORCH
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim



CUDA = torch.cuda.is_available()

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("train_ds", type=str)
    parser.add_argument("-load_path", type=str)
    parser.add_argument("-valid_ds", type=str)
    parser.add_argument("-lr", default=0.001, type=float)
    parser.add_argument("-epochs", default=10, type=int)
    parser.add_argument("-batch_size", default=9, type=int)
    parser.add_argument("-load_memory", action="store_true")
    parser.add_argument("-name", type=str)
    args = parser.parse_args()
    return args

def init_task(args):
    name = args.name if args.name else "Train"
    task: Task = Task.init("Foreground Segmentation", name)
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
    task, logger = init_task(args)
    loader_params = {
        "batch_size": args.batch_size,
        "num_workers": 16,
        "pin_memory": torch.cuda.is_available()
    }
    # define Data
    train_loader, valid_loader = load_datasets(args.train_ds, loader_params, load_memory=args.load_memory, valid_ds_path=args.valid_ds, valid_loader_params=loader_params)
    # define model
    model = net.MobileNetV2_unet()
    if args.load_path: model.load_state_dict(torch.load(args.load_path))
    if CUDA: model.cuda()
    # define Training
    optimizer = optim.Adam(model.params(), lr=args.lr, betas=(0.9, 0.999))
    loss = losses.dice_loss(scale=2)
    trainer  = Trainer(loss, model)
    dict_metrics = {"IOU": metrics.calc_iou}
    # run training and validation
    for E in range(args.epochs):
        mean_loss, mean_metrics = trainer.fit_epoch(train_loader, optimizer, cuda=CUDA, accumulate_gradient=4, metrics=dict_metrics, augment=strong_augment)
        logger.report_scalar("loss", "train", mean_loss, iteration=E)
        for key in mean_metrics:
            logger.report_scalar(key, "train", mean_metrics[key], iteration=E)
        if valid_loader: # if given then run validation
            mean_loss, mean_metrics = trainer.predict(valid_loader, cuda=CUDA, metrics=dict_metrics, augment=just_preprocess)
            logger.report_scalar("loss", "valid", mean_loss, iteration=E)
            for key in mean_metrics:
                logger.report_scalar(key, "valid", mean_metrics[key], iteration=E)
    # saving
    date_s: str = datetime.now().strftime("%d-%m-%y_%H-%M")
    out_dir = join(const.MODEL_DIR, "fg_segmentator", date_s)
    try:
        os.makedirs(out_dir)
    except FileExistsError:
        print("Dir already exists.")
    torch.save(model.state_dict(), join(out_dir, "best.pth"))

if __name__ == "__main__":
    main()