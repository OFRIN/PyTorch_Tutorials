import cv2
import sys
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets, models, transforms

from networks import Generator, Discriminator, initialize_weights

# 1. hyperparameters
image_size = 32
image_channels = 3

batch_size = 128

fake_x_length = 16
fake_y_length = 8

learning_rate = 0.0002

beta1 = 0.5
beta2 = 0.999

latent_dim = 100

max_epoch = 100

# 2. Dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_transforms = transforms.Compose(
    [
        # 0 ~ 255
        transforms.ToTensor(), # 0 ~ 1
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # -1 ~ 1
    ]
)

train_dataset = datasets.CIFAR10('./data/', train=True, download=True, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 3. Networks
G = Generator(image_channels).to(device)
D = Discriminator(image_channels).to(device)

G.apply(initialize_weights)
D.apply(initialize_weights)

loss_fn = nn.BCELoss().to(device)

# 4. Optimizers
optimizer_for_G = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, beta2))
optimizer_for_D = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, beta2))

# 5. Training
fake_gt = np.zeros((batch_size, 1, 1, 1), dtype=np.float32)
fake_gt = torch.FloatTensor(fake_gt).to(device)
fake_gt = torch.autograd.Variable(fake_gt, requires_grad=False)

real_gt = np.ones((batch_size, 1, 1, 1), dtype=np.float32)
real_gt = torch.FloatTensor(real_gt).to(device)
real_gt = torch.autograd.Variable(real_gt, requires_grad=False)

fixed_z = np.random.normal(0, 1, (batch_size, latent_dim, 1, 1))
fixed_z = torch.FloatTensor(fixed_z).to(device)
fixed_z = torch.autograd.Variable(fixed_z, requires_grad=False)

for epoch in range(1, max_epoch + 1):

    G_loss_list = []
    D_loss_list = []

    for x, _ in train_loader:
        x = x.to(device)

        # Generator
        z = np.random.normal(0, 1, (batch_size, latent_dim, 1, 1))
        z = torch.FloatTensor(z).to(device)
        z = torch.autograd.Variable(z, requires_grad=False)

        G_loss = loss_fn(D(G(z)), real_gt)

        optimizer_for_G.zero_grad()
        G_loss.backward()
        optimizer_for_G.step()

        # Discriminator
        D_loss = loss_fn(D(G(z)), fake_gt) + loss_fn(D(x), real_gt)
        D_loss /= 2

        optimizer_for_D.zero_grad()
        D_loss.backward()
        optimizer_for_D.step()

        G_loss_list.append(G_loss.item())
        D_loss_list.append(D_loss.item())

    G_loss = np.mean(G_loss_list)
    D_loss = np.mean(D_loss_list)

    print(f'Epoch={epoch}, G_loss={G_loss}, D_loss={D_loss}')

    # -1 ~ 1 -> 0 ~ 2 -> 0 ~ 255
    # fake_images.shape = (128, 3, 32, 32)
    fake_images = G(fixed_z).cpu().detach().numpy()

    # (128, 3, 32, 32) -> (128, 32, 32, 3)
    # (0,   1,  2,  3) -> (0,    2,  3, 1)
    fake_images = fake_images.transpose((0, 2, 3, 1))

    fake_images = fake_images + 1
    fake_images *= 127.5
    fake_image = []

    for y in range(fake_y_length):
        x_image = fake_images[y * fake_x_length + 0]
        for x in range(1, fake_x_length):
            x_image = np.concatenate([x_image, fake_images[y * 8 + x]], axis=1)

        fake_image.append(x_image.astype(np.uint8))

    fake_image = np.concatenate(fake_image, axis=0)
    cv2.imwrite('./results/%03d.jpg'%epoch, fake_image[..., ::-1])