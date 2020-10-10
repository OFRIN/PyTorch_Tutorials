
import os
import cv2
import glob

import numpy as np

from PIL import Image

from torch.utils.data import DataLoader
from torchvision import models, transforms

from core.data import Single_Classification_Dataset

if __name__ == '__main__':
    root_dir = '../../Toy_Dataset/'

    transforms = transforms.Compose([
        Image.fromarray,
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # normalize 0 ~ 1 and transpose tensor
    ])
    
    dataset = Single_Classification_Dataset(root_dir, transforms, './data/train.json', ['dog', 'cat'])
    loader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=True, pin_memory=True, drop_last=False)
    
    for images, labels in loader:
        
        # torch tensor -> numpy
        images = images.numpy()
        images = images.transpose((0, 2, 3, 1)) * 255
        images = images.astype(np.uint8)

        labels = labels.numpy()

        print(images.shape)
        print(labels.shape)

        for image, label in zip(images, labels):
            print(image.shape, label, np.min(image), np.max(image))
            
            cv2.imshow('show', image)
            cv2.waitKey(0)

