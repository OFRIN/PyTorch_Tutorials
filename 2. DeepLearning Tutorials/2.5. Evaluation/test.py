
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

if __name__ == '__main__':
    os.environ['CUDA_'] = '0,1'

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # 1. Dataset
    root_dir = '../Toy_Dataset/'

    test_transforms = Chain([
        lambda image: cv2.resize(image, (224, 224)),
        Normalize([127.5, 127.5, 127.5], [1.0, 1.0, 1.0]),
        Transpose(),
    ])
    
    test_dataset = Single_Classification_Dataset(root_dir, test_transforms, './data/test.json', ['dog', 'cat'])
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=0, shuffle=False, pin_memory=False, drop_last=False)

    # 2. Network
    model = Customized_VGG16(classes = 2)
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        model.moudle.load_state_dict(torch.load('./model/model.pth'))
    else:
        model = model.to(device)
        model.load_state_dict(torch.load('./model/model.pth'))

    def calculate_accuracy(logits, labels):
        condition = torch.argmax(logits, dim=1) == labels
        accuracy = torch.mean(condition.float())
        return accuracy * 100

    loss_fn = F.cross_entropy
    accuracy_fn = calculate_accuracy

    # 3. Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 4. Training
    model.eval()

    with torch.no_grad():
        test_loss_list = []
        test_accuracy_list = []
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits, predictions = model(images)

            loss = loss_fn(logits, labels)
            accuracy = accuracy_fn(logits, labels)

            test_loss_list.append(loss.item())
            test_accuracy_list.append(accuracy.item())

        test_loss = np.mean(test_loss_list)
        test_accuracy = np.mean(test_accuracy_list)

        print('# test_loss={:.4f}, test_accuracy={:.2f}%'.format(test_loss, test_accuracy))
