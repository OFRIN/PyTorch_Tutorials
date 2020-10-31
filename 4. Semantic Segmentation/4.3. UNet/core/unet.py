

import torch
import torch.nn as nn
import torch.nn.functional as F

class Double_Conv_Block(nn.Module):
    # Conv1->BN1->ReLU1->Conv2->BN2->ReLU2

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()

        self.module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.module(x)

class Down_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        
        self.module = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            Double_Conv_Block(in_channels, mid_channels, out_channels)
        )

    def forward(self, x):
        return self.module(x)

class Up_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()

        self.up_module = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,out_channels=in_channels // 2,kernel_size=2,stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ) 

        self.conv_module = Double_Conv_Block(in_channels, mid_channels, out_channels)

    def forward(self, x1, x2):
        # torch.Size([8, 1024, 14, 14]) 
        # torch.Size([8, 512, 28, 28]) torch.Size([8, 512, 28, 28])
        # torch.Size([8, 1024, 28, 28])

        x1 = self.up_module(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv_module(x)
        return x

class UNet(nn.Module):
    def __init__(self, image_channels, classes, base_features=64):
        super().__init__()

        self.image_channels = image_channels
        self.classes = classes

        self.block = Double_Conv_Block(self.image_channels, base_features, base_features)
        
        self.down_block_1 = Down_Block(base_features, base_features*2, base_features*2)
        self.down_block_2 = Down_Block(base_features*2, base_features*4, base_features*4)
        self.down_block_3 = Down_Block(base_features*4, base_features*8, base_features*8)
        self.down_block_4 = Down_Block(base_features*8, base_features*16, base_features*16)

        self.up_block_1 = Up_Block(base_features*16, base_features*8, base_features*8)
        self.up_block_2 = Up_Block(base_features*8, base_features*4, base_features*4)
        self.up_block_3 = Up_Block(base_features*4, base_features*2, base_features*2)
        self.up_block_4 = Up_Block(base_features*2, base_features, base_features)
        
        self.classifier = nn.Conv2d(in_channels=base_features, out_channels=self.classes, kernel_size=3, stride=1, padding=1)

    def forward(self, images):
        x1 = self.block(images)
        x2 = self.down_block_1(x1)
        x3 = self.down_block_2(x2)
        x4 = self.down_block_3(x3)
        x = self.down_block_4(x4)
        
        x = self.up_block_1(x, x4) # x = [8, 512, 28, 28]
        x = self.up_block_2(x, x3) 
        x = self.up_block_3(x, x2)
        x = self.up_block_4(x, x1)

        return self.classifier(x)

if __name__ == '__main__':
    import numpy as np

    images = np.zeros((8, 1, 224, 224), dtype=np.float32)
    images = torch.from_numpy(images)

    model = UNet(1, 2)

    predictions = model(images)
    predictions = F.softmax(predictions, dim=1)

    print(images.size())
    print(predictions.size())