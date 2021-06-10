import matplotlib.pyplot as plt
import math
from PIL import Image
from torch import Tensor
from torchvision.utils import make_grid, save_image

def plot_history(history):
    fig, axs = plt.subplots(4, 1)
    fig.tight_layout()
    for n, key in enumerate(history):
        data = history[key]
        ax = axs[n]
        ax.plot(data)
        ax.set_title(key)
    return fig

def create_grid(batch: Tensor):
    B, C, H, W = batch.shape
    grid = make_grid(batch, nrow=math.ceil(math.sqrt(B)))
    return grid

def save_batch(batch: Tensor, fp: str):
    grid = create_grid(batch)
    save_image(grid, fp=fp)
