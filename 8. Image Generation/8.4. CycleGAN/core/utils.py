import cv2
import torch
import numpy as np

from PIL import Image

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def convert_OpenCV_to_PIL(data):
    return Image.fromarray(data[..., ::-1])

def convert_PIL_to_OpenCV(data):
    return np.asarray(data)[..., ::-1]
