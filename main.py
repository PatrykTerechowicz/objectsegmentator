import torch
import torch.utils.data as data
import data_loader
import argparse
import os
import model
import torch.nn.functional as F
from datetime import date
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Resize
import sys
is_cuda = torch.cuda.is_available()

preprocess = Compose([Resize(224)])

def load_ds(ds_dir, load_memory, batch_size=9):
    ds = data_loader.ObjectSegmentationDataset(ds_dir=ds_dir, annotation_path=os.path.join(ds_dir, "annotations.json"), load_memory=load_memory, preprocess=preprocess) if ds_dir else None
    if ds: return data.DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("option", default="train", type=str, help="train/test/both")
    parser.add_argument("-train_ds", default=None, type=str)
    parser.add_argument("-test_ds", default=None, type=str)
    parser.add_argument("-valid_ds", default=None, type=str)
    parser.add_argument("-logdir", default=None, type=str)
    parser.add_argument("-lr", default=0.001, type=float)
    parser.add_argument("-epochs", default=10, type=int)
    parser.add_argument("-batch_size", default=9, type=int)
    parser.add_argument("-load_memory", action="store_true")
    args = parser.parse_args()

    option = args.option
    summary = SummaryWriter(os.path.join(args.logdir, date.today().strftime("%Y-%m"))) if args.logdir else None
    print(f"Loading data...")
    load_ds = partial(load_ds, load_memory=args.load_memory, batch_size=args.batch_size)
    train_loader = load_ds(args.train_ds)
    test_loader = load_ds(args.test_ds)
    valid_loader = load_ds(args.valid_ds)
    print(f"Data loaded.")

    net = model.Segmentator()
    net = net.cuda()

    def loss_fn(outputs, targets, eps=1e-15, smooth=1., w=1/3):
        bce = F.binary_cross_entropy(outputs, targets, reduction="mean")
        intersection = (outputs * targets).sum()
        union = outputs.sum() + targets.sum()
        dice = (2. * intersection + smooth)/(union + eps + smooth)
        return w * bce - (1 - w)*torch.log(dice)
    optim = torch.optim.Adam
    print("Starting program.")
    if option == "train":
        net.train(train_loader, valid_loader, epochs=args.epochs, lr=args.lr, summary=summary, loss_fn=loss_fn, optim=optim)
    elif option == "test":
        net.test(test_loader)
    elif option == "both":
        net.train(train_loader, valid_loader, epochs=args.epochs, lr=args.lr, summary=summary)
        net.test(test_loader)
    