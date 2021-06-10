import cv2
import torch
import numpy as np
from model import Segmentator
import argparse
from custom_activations import HardELU

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str, default="best.pth")
args = parser.parse_args()

s = Segmentator(activation_function=HardELU)
saved = torch.load(args.model_path, map_location=torch.device("cpu"))
s.load_state_dict(saved)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    tensor_frame = torch.from_numpy(frame).permute((2, 0, 1)).float()/255.
    C, H, W = tensor_frame.shape
    mask = s(tensor_frame.unsqueeze(0)).squeeze(0)
    frame = tensor_frame*mask
    frame = frame.permute((1, 2, 0))
    frame = frame.numpy()*255.0
    frame = frame.astype(np.uint8)
    cv2.imshow("Obraz", frame)
    if cv2.waitKey(33) == ord('q'):
        break