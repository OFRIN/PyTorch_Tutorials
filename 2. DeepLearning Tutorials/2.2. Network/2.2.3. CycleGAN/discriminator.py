import torch.nn as nn
import torch.nn.functional as F

class Disciminator(nn.Module):
    def __init__(self):
        super().__init__()

        # conv2d (3 -> 64, 4x4, stride=2)
        conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size = 4, stride = 2, padding = 1)
        # LeakyReLU(0.2, inplace=True)
        lrelu1 = nn.LeakyReLU(0.2, True)
        
        # conv2d (64, 128, 4x4, stride=2)
        conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 4, stride = 2, padding = 1)
        # InstanceNorm2d
        insnorm2d2 = nn.InstanceNorm2d(128)
        # LeakyReLU(0.2, inplace=True)
        lrelu2 = nn.LeakyReLU(0.2, True)

        # conv2d (128, 256, 4x4, stride=2)
        conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size = 4, stride = 2, padding = 1)
        # InstanceNorm2d
        insnorm2d3 = nn.InstanceNorm2d(256)
        # LeakyReLU(0.2, inplace=True)
        lrelu3 = nn.LeakyReLU(0.2, True)

        # conv2d (256, 512, 4x4, stride=2)
        conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size = 4, stride = 2, padding = 1)
        # InstanceNorm2d
        insnorm2d4 = nn.InstanceNorm2d(512)
        # LeakyReLU(0.2, inplace=True)  
        lrelu4 = nn.LeakyReLU(0.2, True)

        # conv2d (512, 1, 4x4, stride=2)
        conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size = 4, stride = 2, padding = 1)

        self.model = nn.Sequential(*[
            conv1,
            lrelu1, 
            
            conv2, 
            insnorm2d2,
            lrelu2,
            
            conv3, 
            insnorm2d3, 
            lrelu3, 
            
            conv4, 
            insnorm2d4, 
            lrelu4, 
            
            conv5
        ])
        
    def GAP(self, x):
        b, c, h, w = x.size()

        # 16, 512, 7, 7 -> 16, 512, 1, 1
        x = F.avg_pool2d(x, [h, w])

        # 16, 512, 1, 1 -> 16, 512
        x = x.view(b, c)

        return x

    def forward(self, x):
        features = self.model(x)
        logits = self.GAP(features)
        return logits

