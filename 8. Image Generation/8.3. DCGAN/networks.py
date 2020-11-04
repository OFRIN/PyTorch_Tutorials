import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(w):
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, image_channels):
        super().__init__()

        self.image_channels = image_channels
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, kernel_size=4, stride=1, padding = 0, bias = False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True), # 1, 1024, 4, 4

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True), # 1, 512, 8, 8

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True), # 1, 256, 16, 16

            nn.ConvTranspose2d(256, self.image_channels, kernel_size=4, stride=2, padding = 1, bias = False),

            nn.Tanh(), # -1 ~ 1, (1, 3, 32, 32)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, image_channels):
        super().__init__()

        self.image_channels = image_channels
        self.model = nn.Sequential(
            # (1, 3, 32, 32) -> (1,64,16,16)
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding = 1, bias=False),
            nn.LeakyReLU(0.2, True),

            # (1, 64, 16, 16) -> (1,128,8,8)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding = 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            # (1, 128, 8, 8) -> (1,256,4,4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding = 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # (1, 256, 4, 4) -> (1, 1, 1, 1) -> (1, 1)
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding = 0, bias=False),

            nn.Sigmoid()
        )

    def forward(self, images):
        return self.model(images)

if __name__ == '__main__':
    G = Generator(3)
    D = Discriminator(3)

    import numpy as np
    z = np.random.normal(0, 1, (16, 100, 1, 1))
    z = torch.FloatTensor(z)

    fake_images = G(z)
    fake_preds = D(G(z))

    print(z.size())
    print(fake_images.size())
    print(fake_preds.size())