import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms

from torch.utils.data import DataLoader
from utility.utils import convert_OpenCV_to_PIL
from core.data import ISBI_Dataset_For_Testing
from core.unet import UNet
from core.utils import get_numpy_from_tensor

from utility.utils import transpose

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    image_dir = 'C:/DB/ISBI/'
    class_names = ['cell']

    batch_size = 1
    image_size = 256
    use_cores = 1

    test_transforms = transforms.Compose(
            [
                convert_OpenCV_to_PIL,
                # transforms.Resize(image_size),
                transforms.ToTensor(), # / 255
            ]
        )

    dataset = ISBI_Dataset_For_Testing(image_dir + 'test-volume.tif', test_transforms, class_names)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=use_cores, pin_memory=False, shuffle=True, drop_last=True)

    model = UNet(1,1,base_features=16).to(device)
    model.eval()
    
    model.load_state_dict(torch.load('./model/unet.pth'))

    import cv2 

    for images in loader:
        images=images.to(device)

        logits = model(images)
        predictions = F.sigmoid(logits)

        # Test
        images = get_numpy_from_tensor(images)
        predictions = get_numpy_from_tensor(predictions)

        # B, C, H, W -> b,h,w,c
        # 0, 1, 2, 3 -> 0,2,3,1

        images = transpose(images, (0, 2, 3, 1))
        predictions = transpose(predictions, (0, 2, 3, 1))
        
        for i in range(batch_size):
            image = images[i] # 0~1 -> 0~255
            prediction = predictions[i] #0~1 -> 0~255

            image =image*255
            prediction = prediction*255 

            image = image.astype('uint8')
            prediction = prediction.astype('uint8')

            cam = cv2.applyColorMap(prediction, cv2.COLORMAP_HOT)

            cv2.imshow('img',image)
            cv2.imshow('predction',prediction)
            cv2.imshow('cam',cam)
            cv2.waitKey(0)