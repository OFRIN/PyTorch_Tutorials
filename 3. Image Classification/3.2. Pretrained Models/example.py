import cv2
import torch

import numpy as np

from torchvision import models, transforms

from utility.utils import convert_OpenCV_to_PIL

test_transforms = transforms.Compose([
    convert_OpenCV_to_PIL,

    transforms.Resize(224),
    # transforms.CenterCrop(224),

    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# model = models.mobilenet_v2(pretrained=True)
model = models.resnet18(pretrained=True)
model.eval()

image = cv2.imread('./test_images/giant_panda.jpg')
cv2.imshow('show', image)
cv2.waitKey(0)

image_tensor = torch.unsqueeze(test_transforms(image), 0)
predictions = model(image_tensor)
predictions = predictions[0].cpu().detach().numpy()

indices = np.argsort(predictions)[::-1]
top5_class_indices = indices[:5]

print(top5_class_indices)

