
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
    parser.add_argument('--init_lr', default=1e-3, type=float)
    
    parser.add_argument('--batch_size', default=32, type=int)
    
    parser.add_argument('--max_epochs', default=10, type=int)
    
    ###############################################################################
    # Training Technology
    ###############################################################################
    parser.add_argument('--architecture', default='resnet-18', type=str)
    parser.add_argument('--model_name', default='resnet-18', type=str)
    parser.add_argument('--pretrained', default=True, type=str2bool)
    
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

    transforms_dic = {
        'train' : transforms.Compose([
            convert_OpenCV_to_PIL,

            transforms.Resize(256),
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

    data_dir = args.root_dir + 'dataset/'
    class_names = [line.strip() for line in open(args.root_dir + 'class.names').readlines()]
    
    dataset_dic = {domain : Single_Classification_Dataset(data_dir + domain, transforms_dic[domain], args.root_dir + '{}.json'.format(domain), class_names) for domain in ['train', 'validation']}
    loader_dic = {
        domain : DataLoader(
            dataset_dic[domain], batch_size=args.batch_size, num_workers=args.use_cores, pin_memory=False,
            shuffle=True if domain == 'train' else False, 
            drop_last=True if domain == 'train' else False 
        ) for domain in ['train', 'validation']
    }
    
    # 2. Network
    if args.architecture == 'resnet-18':
        model = models.resnet18(pretrained=args.pretrained)
        model.fc = nn.Linear(model.fc.in_features, dataset_dic['train'].classes)
        
    elif args.architecture == 'densenet-161':
        model = models.densenet161(pretrained=args.pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, dataset_dic['train'].classes)
    
    if use_cuda:
        model = model.to(device)
        load_model_fn = lambda: model.state_dict(torch.load(model_path))
        save_model_fn = lambda: torch.save(model.state_dict(), model_path)
    else:
        model = torch.nn.DataParallel(model).cuda()
        load_model_fn = lambda: model.module.state_dict(torch.load(model_path))
        save_model_fn = lambda: torch.save(model.module.state_dict(), model_path)
    
    def calculate_accuracy(logits, labels):
        condition = torch.argmax(logits, dim=1) == labels
        accuracy = torch.mean(condition.float())
        return accuracy * 100

    loss_fn = F.cross_entropy
    accuracy_fn = calculate_accuracy

    # 3. Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9)
    
    # 4. Training
    writer = SummaryWriter(f'./experiments/tensorboard/{args.model_name}')

    max_epochs = args.max_epochs
    best_valid_accuacy = 0.0

    csv_print(['Epoch', 'Phase', 'Loss', 'Accuracy'], csv_path)

    for epoch in range(1, max_epochs + 1):

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
                
                logits = model(images)

                loss = loss_fn(logits, labels)
                accuracy = accuracy_fn(logits, labels)

                # print(images.size(), labels.size(), loss.item(), accuracy.item())

                loss_list.append(loss.item())
                accuracy_list.append(accuracy.item())

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            avg_loss = np.mean(loss_list)
            avg_accuracy = np.mean(accuracy_list)

            writer.add_scalar(f'{phase}/loss', avg_loss, epoch)
            writer.add_scalar(f'{phase}/accuracy', avg_accuracy, epoch)
            
            print('# epoch={}, phase={}, loss={:.4f}, accuracy={:.2f}%'.format(epoch, phase, avg_loss, avg_accuracy))
            csv_print([epoch, phase, avg_loss, avg_accuracy], csv_path)

            if phase == 'validation' and best_valid_accuacy < avg_accuracy:
                best_valid_accuacy = avg_accuracy
                save_model_fn()

    writer.close()

