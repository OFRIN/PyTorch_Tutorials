import torch
from torch import nn
from torch.nn import functional as F

class Customized_VGG16_with_BN(nn.Module):
    def __init__(self, classes):
        super().__init__()

        self.classes = classes

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(16)

        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(16)

        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2_1 = nn.Conv2d(16,32,3,stride=1, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(32)

        self.conv2_2 = nn.Conv2d(32,32,3,stride=1, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(32)

        self.pool2 = nn.MaxPool2d((2, 2), stride=2)

        self.conv3_1 = nn.Conv2d(32,64,3,stride=1, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(64)
        
        self.conv3_2 = nn.Conv2d(64,64,3,stride=1, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(64)

        self.conv3_3 = nn.Conv2d(64,64,3,stride=1, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(64)

        self.pool3 = nn.MaxPool2d((2, 2), stride=2)

        self.conv4_1 = nn.Conv2d(64,128,3,stride=1, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(128)

        self.conv4_2 = nn.Conv2d(128,128,3,stride=1, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(128)

        self.conv4_3 = nn.Conv2d(128,128,3,stride=1, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(128)

        self.pool4 = nn.MaxPool2d((2, 2), stride=2)

        self.flatten = nn.Flatten()
        
        self.dense1 = nn.Linear(128 * 14 * 14, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, self.classes)

    def forward(self, images):
        x = images

        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
        x = F.relu(self.conv3_3_bn(self.conv3_3(x)))
        x = self.pool3(x)

        x = F.relu(self.conv4_1_bn(self.conv4_1(x)))
        x = F.relu(self.conv4_2_bn(self.conv4_2(x)))
        x = F.relu(self.conv4_3_bn(self.conv4_3(x)))
        x = self.pool4(x)
        
        x = self.flatten(x)
        
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))

        logits = self.dense3(x)
        preds = F.softmax(logits, dim=1)

        return logits, preds

class Customized_VGG16(nn.Module):
    def __init__(self, classes):
        super().__init__()

        self.classes = classes

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2_1 = nn.Conv2d(16,32,3,stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(32,32,3,stride=1, padding=1)
        self.pool2 = nn.MaxPool2d((2, 2), stride=2)

        self.conv3_1 = nn.Conv2d(32,64,3,stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(64,64,3,stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(64,64,3,stride=1, padding=1)
        self.pool3 = nn.MaxPool2d((2, 2), stride=2)

        self.conv4_1 = nn.Conv2d(64,128,3,stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(128,128,3,stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(128,128,3,stride=1, padding=1)
        self.pool4 = nn.MaxPool2d((2, 2), stride=2)

        self.flatten = nn.Flatten()
        
        self.dense1 = nn.Linear(128 * 14 * 14, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, self.classes)

    def forward(self, images):
        x = images

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)
        
        x = self.flatten(x)
        
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))

        logits = self.dense3(x)
        preds = F.softmax(logits, dim=1)

        return logits, preds

class VGG16(nn.Module):
    def __init__(self, classes):
        super().__init__()

        self.classes = classes

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2_1 = nn.Conv2d(64,128,3,stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128,128,3,stride=1, padding=1)
        self.pool2 = nn.MaxPool2d((2, 2), stride=2)

        self.conv3_1 = nn.Conv2d(128,256,3,stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256,256,3,stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256,256,3,stride=1, padding=1)
        self.pool3 = nn.MaxPool2d((2, 2), stride=2)

        self.conv4_1 = nn.Conv2d(256,512,3,stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512,512,3,stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512,512,3,stride=1, padding=1)
        self.pool4 = nn.MaxPool2d((2, 2), stride=2)

        self.conv5_1 = nn.Conv2d(512,512,3,stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512,512,3,stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512,512,3,stride=1, padding=1)
        self.pool5 = nn.MaxPool2d((2, 2), stride=2)

        self.flatten = nn.Flatten()
        
        self.dense1 = nn.Linear(512 * 7 * 7, 4096)
        self.dense2 = nn.Linear(4096, 4096)
        self.dense3 = nn.Linear(4096, self.classes)

    def forward(self, images):
        x = images

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)

        x = self.flatten(x)

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))

        logits = self.dense3(x)
        preds = F.softmax(logits, dim=1)

        return logits, preds

