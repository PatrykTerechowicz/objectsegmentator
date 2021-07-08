from tqdm import tqdm
from typing import Optional, Callable, List, Dict
import torch

def iterative_mean(mean, new_value, iter):
    return (mean*iter + new_value)/(iter+1)

class Trainer:
    def __init__(self, loss, model):
        self.loss = loss
        self.model = model
    
    def fit_epoch(self, loader, optimizer, accumulate_gradient: int=1, cuda: bool=True, metrics: Dict={}, augment: Optional[Callable] = None):
        mean_loss = 0
        metrics_mean = dict.fromkeys(metrics.keys(), 0)
        for iter, data in tqdm(enumerate(loader), desc=f"training", total=len(loader)):
            image, target = data
            if augment: image, target = augment(image, target)
            if cuda:
                image = image.cuda()
                target = target.cuda()
            output = self.model(image)
            loss_value = self.loss(output, target)
            loss_value.backward()
            if iter % accumulate_gradient == 0:
                optimizer.step()
                optimizer.zero_grad()
            mean_loss = iterative_mean(mean_loss, loss_value, iter)
            for key in metrics_mean:
                mean = metrics_mean[key]
                metric_value = metrics[key](output, target)
                metrics_mean[key] = iterative_mean(mean, metric_value, iter)
        return mean_loss, metrics_mean

    @torch.no_grad()
    def predict(self, loader, cuda: bool=True, metrics: Dict={}, augment: Optional[Callable] = None, save_results=None):
        mean_loss = 0
        metrics_mean = dict.fromkeys(metrics.keys(), 0)
        for iter, data in tqdm(enumerate(loader), desc=f"making_prediction", total=len(loader)):
            image, target = data
            if augment: image, target = augment(image, target)
            if cuda:
                image = image.cuda()
                target = target.cuda()
            output = self.model(image)
            loss_value = self.loss(output, target)
            mean_loss = iterative_mean(mean_loss, loss_value, iter)
            for key in metrics_mean:
                mean = metrics_mean[key]
                metric_value = metrics[key](output, target)
                metrics_mean[key] = iterative_mean(mean, metric_value, iter)
        return mean_loss, metrics_mean
