import torch
from torch import nn
from torch.nn import functional as F

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

        x = F.relu(F.dropout(self.dense1(x), p=0.5, training=True))
        x = F.relu(F.dropout(self.dense2(x), p=0.5, training=True))
        logits = self.dense3(x)
        preds = F.softmax(logits, dim=1)

        return logits, preds

if __name__ == '__main__':
    model = VGG16(10)

    images = torch.randn(16, 3, 224, 224)
    logits, preds = model.forward(images)

    print(images.size())
    print(preds.size())