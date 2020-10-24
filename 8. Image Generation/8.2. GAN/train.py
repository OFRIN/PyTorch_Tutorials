import cv2
import sys
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets, models, transforms

from networks import Generator, Discriminator

# 1. hyperparameters
image_size = 28
batch_size = 64
learning_rate = 0.0002

beta1 = 0.5
beta2 = 0.999

latent_dim = 128

max_epoch = 200

# 2. Dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

train_dataset = datasets.MNIST('./data/', train=True, download=True, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 3. Networks
G = Generator(image_size).to(device)
D = Discriminator(image_size).to(device)

loss_fn = nn.BCELoss().to(device)

# 4. Optimizers
optimizer_for_G = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, beta2))
optimizer_for_D = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, beta2))

# 5. Training
fake_gt = np.zeros((batch_size, 1), dtype=np.float32)
fake_gt = torch.FloatTensor(fake_gt).to(device)
fake_gt = torch.autograd.Variable(fake_gt, requires_grad=False)

real_gt = np.ones((batch_size, 1), dtype=np.float32)
real_gt = torch.FloatTensor(real_gt).to(device)
real_gt = torch.autograd.Variable(real_gt, requires_grad=False)

for epoch in range(1, max_epoch + 1):

    G_loss_list = []
    D_loss_list = []

    for x, _ in train_loader:
        x = x.to(device)

        # Generator
        z = np.random.normal(0, 1, (batch_size, latent_dim))
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

    