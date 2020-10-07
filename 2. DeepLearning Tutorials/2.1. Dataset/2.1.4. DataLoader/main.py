
import os
import cv2
import glob

from torch.utils.data import DataLoader

from data import Single_Classification_Dataset

from augmentation.augment_utils import Chain
from augmentation.augment_utils import Random_HorizontalFlip
from augmentation.augment_utils import Random_ColorJitter

root_dir = '../../Toy_Dataset/'

transforms = Chain([
    Random_HorizontalFlip(0.5),
    Random_ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1),
    lambda image: cv2.resize(image, (224, 224))
])

dataset = Single_Classification_Dataset(root_dir, transforms, './data/train.json', ['dog', 'cat'])
loader = DataLoader(dataset, batch_size=16, num_workers=2, shuffle=True, pin_memory=True, drop_last=False)

for images, labels in loader:

    # torch tensor -> numpy
    images = images.numpy()
    labels = labels.numpy()

    for image, label in zip(images, labels):
        print(image.shape, label)
        
        cv2.imshow('show', image)
        cv2.waitKey(0)

