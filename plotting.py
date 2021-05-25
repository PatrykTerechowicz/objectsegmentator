import matplotlib.pyplot as plt
import math
from PIL import Image
from torch import Tensor
from torchvision.utils import make_grid, save_image

def plot_history(history):
    fig, axs = plt.subplots(2, 2)
    for n, key in enumerate(history):
        data = history[key]
        ax = axs[n]
        ax.plot(data)
        ax.title(key)
    return fig

def save_batch(batch: Tensor, fp: str):
    B, C, H, W = batch.shape
    grid = make_grid(batch, nrow=math.ceil(math.sqrt(B)))
    save_image(grid, fp=fp)