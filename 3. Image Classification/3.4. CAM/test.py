
import os
import cv2
import glob
import argparse

import numpy as np
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import models, transforms

from PIL import Image

from core.data import Single_Classification_Dataset
from utility.utils import convert_OpenCV_to_PIL, csv_print

from core.utils import get_numpy_from_tensor

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config():
    parser = argparse.ArgumentParser()
    
    ###############################################################################
    # GPU Config
    ###############################################################################
    parser.add_argument('--use_gpu', default='0', type=str)
    
    ###############################################################################
    # Dataset
    ###############################################################################
    parser.add_argument('--use_cores', default=mp.cpu_count(), type=int)
    parser.add_argument('--root_dir', default='C:/DB/Animals-10/', type=str)

    ###############################################################################
    # Training Schedule
    ###############################################################################
    parser.add_argument('--batch_size', default=32, type=int)
    
    ###############################################################################
    # Training Technology
    ###############################################################################
    parser.add_argument('--architecture', default='resnet-18', type=str)
    parser.add_argument('--model_name', default='resnet-18', type=str)

    return parser.parse_args()

if __name__ == '__main__':

    args = get_config()
    use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # 1. Dataset
    model_dir = './experiments/model/'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model_path = model_dir + '{}.pth'.format(args.model_name)
    csv_path = './experiments/model/' + '{}.csv'.format(args.model_name)

    data_dir = args.root_dir + 'dataset/'
    class_names = [line.strip() for line in open(args.root_dir + 'class.names').readlines()]

    domain = 'train'

    test_transforms = transforms.Compose([
        convert_OpenCV_to_PIL,

        transforms.Resize(256),
        transforms.CenterCrop(224),
        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    test_dataset = Single_Classification_Dataset(data_dir + domain, test_transforms, args.root_dir + '{}.json'.format(domain), class_names)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.use_cores, pin_memory=False, shuffle=True, drop_last=False)

    # 2. Network
    if args.architecture == 'resnet-18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, test_dataset.classes)
        
    elif args.architecture == 'densenet-161':
        model = models.densenet161(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, test_dataset.classes)
    
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    
    # 3. Evaluation
    model.eval()
    
    with torch.no_grad():

        accuracy_list = []
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            
            accuracy = torch.mean((predictions == labels).float())
            accuracy_list.append(accuracy.item())

        test_accuracy = np.mean(accuracy_list)
        print('Accuracy = {:.2f}%'.format(test_accuracy * 100))
