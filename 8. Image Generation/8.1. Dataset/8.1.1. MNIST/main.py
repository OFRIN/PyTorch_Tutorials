import cv2
import torch
import numpy as np

from torchvision import datasets, models

train_dataset = datasets.MNIST('./data/', train=True, download=True)
test_dataset = datasets.MNIST('./data/', train=False, download=True)

print(len(train_dataset)) # 60.000
print(len(test_dataset)) # 10,000

print(train_dataset[0]) # image, label

for i in range(len(train_dataset)):
    image, label = train_dataset[i]
    image = np.asarray(image, dtype=np.uint8)
    image = cv2.resize(image, (224, 224))
    cv2.imshow('show', image)
    cv2.waitKey(0)