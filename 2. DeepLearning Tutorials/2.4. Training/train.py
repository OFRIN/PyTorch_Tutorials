
import os
import cv2
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.utils.data import DataLoader

from core.vgg16 import VGG16, Customized_VGG16
from core.data import Single_Classification_Dataset

from core.augmentation.augment_utils import Chain
from core.augmentation.augment_utils import Random_HorizontalFlip
from core.augmentation.augment_utils import Random_ColorJitter
from core.augmentation.augment_utils import Normalize
from core.augmentation.augment_utils import Transpose

# 1. Dataset
root_dir = '../Toy_Dataset/'

train_transforms = Chain([
    Random_HorizontalFlip(0.5),
    Random_ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1),
    lambda image: cv2.resize(image, (224, 224)),
    Normalize([127.5, 127.5, 127.5], [1.0, 1.0, 1.0]),
    Transpose(),
])

test_transforms = Chain([
    lambda image: cv2.resize(image, (224, 224)),
    Normalize([127.5, 127.5, 127.5], [1.0, 1.0, 1.0]),
    Transpose(),
])

train_dataset = Single_Classification_Dataset(root_dir, train_transforms, './data/train.json', ['dog', 'cat'])
train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

valid_dataset = Single_Classification_Dataset(root_dir, test_transforms, './data/validation.json', ['dog', 'cat'])
valid_loader = DataLoader(valid_dataset, batch_size=16, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)

# 2. Network
model = Customized_VGG16(classes = 2)
model = torch.nn.DataParallel(model).cuda()

def calculate_accuracy(logits, labels):
    condition = torch.argmax(logits, dim=1) == labels
    accuracy = torch.mean(condition.float())
    return accuracy * 100

loss_fn = F.cross_entropy
accuracy_fn = calculate_accuracy

# 3. Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 4. Training
max_epochs = 100

for epoch in range(max_epochs):

    model.train()

    train_loss_list = []
    train_accuracy_list = []

    for images, labels in train_loader:
        images = images.float().cuda()
        labels = labels.cuda()
        
        logits, predictions = model(images)

        loss = loss_fn(logits, labels)
        accuracy = accuracy_fn(logits, labels)

        train_loss_list.append(loss.item())
        train_accuracy_list.append(accuracy.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = np.mean(train_loss_list)
    train_accuracy = np.mean(train_accuracy_list)
    
    print('# epoch={}, loss={:.4f}, train_accuracy={:.2f}%'.format(epoch + 1, train_loss, train_accuracy))

    torch.save(model.module.state_dict(), './model/model.pth')