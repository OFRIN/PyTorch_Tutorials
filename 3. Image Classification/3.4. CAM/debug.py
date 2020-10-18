import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from pytorch_model_summary import summary

from torchvision import models, transforms

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

print(model.fc.weight.size())

print(model.fc.weight[0].size())
print(model.fc.weight[1].size())

# print(model.layer4)

print(summary(model, torch.zeros((1, 3, 224, 224)), show_input=True, show_hierarchical=True))

# Conv = [7, 7, 512]
# Weights = [512]
# Conv * Weights => CAM

image_tensor = torch.zeros((1, 3, 224, 224))

extractor = nn.Sequential(*[
    model.conv1,
    model.bn1,
    model.relu,
    model.maxpool,

    model.layer1,
    model.layer2,
    model.layer3,
    model.layer4,
])

features = extractor(image_tensor)

fc = nn.Conv2d(512,2,(1,1))

cam = fc(features)

print(features.size())     # 1, 512, 7, 7
print(cam.size()) # 1, 2, 7, 7

logits = F.avg_pool2d(cam, kernel_size=(7, 7), padding=0)

