
import os
import cv2
import glob

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import models, transforms

from PIL import Image

from core.vgg16 import VGG16, Customized_VGG16, Customized_VGG16_with_BN
from core.data import Single_Classification_Dataset
from utility.utils import convert_OpenCV_to_PIL

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # 1. Dataset
    root_dir = '../Toy_Dataset/'

    model_dir = './model/'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    transforms_dic = {
        'train' : transforms.Compose([
            convert_OpenCV_to_PIL,

            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1),
            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'validation' : transforms.Compose([
            convert_OpenCV_to_PIL,

            transforms.Resize(256),
            transforms.CenterCrop(224),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }

    dataset_dic = {domain : Single_Classification_Dataset(root_dir, transforms_dic[domain], f'./data/{domain}.json', ['dog', 'cat']) for domain in ['train', 'validation']}
    loader_dic = {
        domain : DataLoader(
            dataset_dic[domain], batch_size=32, num_workers=0, pin_memory=False,
            shuffle=True if domain == 'train' else False, 
            drop_last=True if domain == 'train' else False 
        ) for domain in ['train', 'validation']
    }
    
    # 2. Network
    model = Customized_VGG16_with_BN(classes = 2)
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        load_model_fn = lambda model_path: model.module.state_dict(torch.load(model_path))
        save_model_fn = lambda model_path: torch.save(model.module.state_dict(), model_path)

    else:
        model = model.to(device)
        load_model_fn = lambda model_path: model.state_dict(torch.load(model_path))
        save_model_fn = lambda model_path: torch.save(model.state_dict(), model_path)

    def calculate_accuracy(logits, labels):
        condition = torch.argmax(logits, dim=1) == labels
        accuracy = torch.mean(condition.float())
        return accuracy * 100

    loss_fn = F.cross_entropy
    accuracy_fn = calculate_accuracy

    # 3. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 4. Training
    max_epochs = 200
    best_valid_accuacy = 0.0

    for epoch in range(max_epochs):

        for phase in ['train', 'validation']:
            if phase == 'train': 
                model.train()
            else: 
                model.eval()
        
            loss_list = []
            accuracy_list = []

            for images, labels in loader_dic[phase]:
                images = images.to(device)
                labels = labels.to(device)
                
                logits, predictions = model(images)

                loss = loss_fn(logits, labels)
                accuracy = accuracy_fn(logits, labels)

                loss_list.append(loss.item())
                accuracy_list.append(accuracy.item())

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            avg_loss = np.mean(loss_list)
            avg_accuracy = np.mean(accuracy_list)
            
            print('# epoch={}, phase={}, loss={:.4f}, accuracy={:.2f}%'.format(epoch + 1, phase, avg_loss, avg_accuracy))

            if phase == 'validation' and best_valid_accuacy < avg_accuracy:
                best_valid_accuacy = avg_accuracy
                save_model_fn('./model/model.pth')



