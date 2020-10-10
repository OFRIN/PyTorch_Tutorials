
import os
import cv2
import glob

from torch.utils.data import DataLoader
from opencv_transforms import transforms

from core.data import Single_Classification_Dataset

if __name__ == '__main__':
    root_dir = '../../Toy_Dataset/'

    transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1),
        transforms.RandomHorizontalFlip(),
        lambda image: cv2.resize(image, (224, 224))
    ])

    dataset = Single_Classification_Dataset(root_dir, transforms, './data/train.json', ['dog', 'cat'])
    loader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=True, pin_memory=True, drop_last=False)
    
    for images, labels in loader:

        # torch tensor -> numpy
        images = images.numpy()
        labels = labels.numpy()

        print(images.shape)
        print(labels.shape)

        for image, label in zip(images, labels):
            print(image.shape, label)
            
            cv2.imshow('show', image)
            cv2.waitKey(0)

