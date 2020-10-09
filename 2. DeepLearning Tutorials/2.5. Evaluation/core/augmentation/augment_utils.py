import cv2
import random

import numpy as np

from core.augmentation.utils import hflip
from core.augmentation.utils import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue

class Chain:
    def __init__(self, augment_functions = []):
        self.augment_functions = augment_functions
    
    def __call__(self, images):
        aug_images = images.copy()
        for augment_func in self.augment_functions:
            aug_images = augment_func(aug_images)
        return aug_images

class Transpose:
    def __init__(self):
        pass

    def __call__(self, data):

        # D, H, W, C -> C, D, H, W
        if len(data.shape) == 4:
           data = data.transpose((3, 0, 1, 2)) 

        # H, W, C -> C, H, W
        else:
            data = data.transpose((2, 0, 1))

        return data

class Normalize:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, images):
        return ((images - self.mean) / self.std).astype(np.float32)

class Random_HorizontalFlip:
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return hflip(x)
        return x

class Random_ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, x):
        transforms = []

        if self.brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            transforms.append(lambda img: adjust_brightness(img, brightness_factor))

        if self.contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            transforms.append(lambda img: adjust_contrast(img, contrast_factor))

        if self.saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            transforms.append(lambda img: adjust_saturation(img, saturation_factor))

        if self.hue > 0:
            hue_factor = random.uniform(-self.hue, self.hue)
            transforms.append(lambda img: adjust_hue(img, hue_factor))

        random.shuffle(transforms)
        
        for transform in transforms:
            x = transform(x)

        return x

