import torch
import torch.utils.data as data
import data_loader
import argparse
import os
import model
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import Compose, ToTensor, Resize

is_cuda = torch.cuda.is_available()

transform = Compose([Resize(224), ToTensor()])

def load_ds(ds_dir, load_memory, batch_size=9):
    ds = data_loader.ObjectSegmentationDataset(ds_dir=ds_dir, annotation_path=os.path.join(ds_dir, "annotations.json"), load_memory=load_memory, transform=transform) if ds_dir else None
    if ds: return data.DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("option", default="train", type=str, help="train/test/both")
    parser.add_argument("-train_ds", default=None, type=str)
    parser.add_argument("-test_ds", default=None, type=str)
    parser.add_argument("-valid_ds", default=None, type=str)
    parser.add_argument("-logdir", default=None, type=str)
    parser.add_argument("-load_memory", action="store_true")
    args = parser.parse_args()

    option = args.option
    summary = SummaryWriter(args.logdir) if args.logdir else None
    load_ds = partial(load_ds, load_memory=args.load_memory)
    train_loader = load_ds(args.train_ds)
    test_loader = load_ds(args.test_ds)
    valid_loader = load_ds(args.valid_ds)

    net = model.Segmentator()
    net = net.cuda()
    print("Loaded stuff.\n Starting program.")
    if option == "train":
        net.train(train_loader, valid_loader, epochs=10, lr=0.001, summary=summary)
    elif option == "test":
        pass
    elif option == "both":
        pass
    