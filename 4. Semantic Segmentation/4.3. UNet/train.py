import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms

from torch.utils.data import DataLoader
from utility.utils import convert_OpenCV_to_PIL
from core.data import ISBI_Dataset_For_Training
from core.unet import UNet

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    image_dir = 'C:/DB/ISBI/'
    class_names = ['cell']

    batch_size = 4
    image_size = 256
    use_cores = 1

    test_transforms = transforms.Compose(
        [
            convert_OpenCV_to_PIL,
            # transforms.Resize(image_size),
            transforms.ToTensor(), # / 255
        ]
    )

    dataset = ISBI_Dataset_For_Training(image_dir, test_transforms, class_names)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=use_cores, pin_memory=False, shuffle=True, drop_last=True)

    # for images, labels in loader:
    #   print(images.size(), labels.size())
    # print(np.unique(labels))

    # Network

    # 1. Background, Cell (classes=2, softmax) background=40%, cell=60%
    # 2. Cell (classes=1, sigmoid) cell=60%, background=40%
    model = UNet(1,1,base_features=16).to(device)

    # Loss
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training (Epoch)

    epochs = 100
    for epoch in range(1, epochs+1):

        train_loss_list = []
        
        for images, labels in loader:
            images=images.to(device)
            labels=labels.to(device)
            
            logits = model(images)

            loss=loss_fn(logits,labels)

            optimizer.zero_grad() # 초기화
            loss.backward() 
            optimizer.step() # 한 step 이동

            train_loss_list.append(loss.item()) # tensor에서 scalar 값 뽑는 item

        mean_train_loss = np.mean(train_loss_list)
        print(f'epoch={epoch}, loss={mean_train_loss}')
        
        torch.save(model.state_dict(), './model/unet.pth')
        # model.load_state_dict(torch.load('./model/unet.pth'))
