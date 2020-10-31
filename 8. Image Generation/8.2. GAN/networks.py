import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.image_size = image_size
        self.model = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=self.image_size**2),
            nn.Tanh(), # -1 ~ 1
        )

    def forward(self, z):
        images = self.model(z)
        images = images.view(z.size(0), self.image_size, self.image_size)
        return images

class Discriminator(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.image_size = image_size
        self.model = nn.Sequential(
            nn.Linear(in_features=self.image_size**2, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, images):
        images = images.view(images.size(0), self.image_size**2)
        predictions = self.model(images)
        return predictions

if __name__ == '__main__':
    G = Generator(28)
    D = Discriminator(28)

    import numpy as np
    z = torch.FloatTensor(np.random.normal(0, 1, (16, 128)))

    fake_images = G(z)
    fake_preds = D(G(z))

    fake_images = fake_images.view((-1, 28, 28))

    print(z.size())
    print(fake_images.size())
    print(fake_preds.size(), fake_preds)