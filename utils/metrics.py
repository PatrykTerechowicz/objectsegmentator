import torch

def calc_iou(out_mask, true_mask):
    intersection = ( out_mask > .5 ) & (true_mask > .9)
    intersection = torch.sum(intersection.float())
    union = torch.sum(out_mask) + torch.sum(true_mask) - intersection
    return intersection/union