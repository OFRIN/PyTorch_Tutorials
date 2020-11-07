
import cv2
import torch
import itertools

from PIL import Image
from torchvision import transforms

from core.dataset import ImageDataset
from core.networks import Generator, Discriminator
from core.utils import *

# 1. Define
root_dir = './datasets/horse2zebra/'
learning_rate = 0.0002
batch_size = 1
image_size = 256
A_channels = 3
B_channels = 3
max_epoch = 200

# 2. Dataset
transform = transforms.Compose(
    [
        convert_OpenCV_to_PIL, 

        transforms.Resize(int(image_size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),

        # convert_PIL_to_OpenCV

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

A_dataset = ImageDataset(root_dir, transform=transform, mode='trainA')
B_dataset = ImageDataset(root_dir, transform=transform, mode='trainB')

A_loader = torch.utils.data.DataLoader(A_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
B_loader = torch.utils.data.DataLoader(B_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# for i in range(len(A_dataset)):
#     cv2.imshow('A', A_dataset[i])
#     cv2.imshow('B', B_dataset[i])
#     cv2.waitKey(0)

# 3. Train
G_A2B = Generator(A_channels, B_channels, 1).cuda()
G_B2A = Generator(B_channels, A_channels, 1).cuda()
D_A = Discriminator(A_channels).cuda()
D_B = Discriminator(B_channels).cuda()

G_A2B.apply(weights_init_normal)
G_B2A.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

# 4. Loss
GAN_loss_fn = torch.nn.MSELoss().cuda()
cycle_loss_fn = torch.nn.L1Loss().cuda()
identity_loss_fn = torch.nn.L1Loss().cuda()

# 5. Optimizer
optimizer_G = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# 6. Training
# input_A = torch.Tensor(batch_size, A_channels, image_size, image_size)
# input_B = torch.Tensor(batch_size, B_channels, image_size, image_size)
real_labels = torch.autograd.Variable(torch.Tensor(batch_size).fill_(1.0), requires_grad=False).cuda()
fake_labels = torch.autograd.Variable(torch.Tensor(batch_size).fill_(0.0), requires_grad=False).cuda()

for epoch in range(1, max_epoch + 1):
    for real_A_images, real_B_images in zip(A_loader, B_loader):
        real_A_images = real_A_images.cuda()
        real_B_images = real_B_images.cuda()
        
        fake_B_images = G_A2B(real_A_images)
        real_B_logits = D_B(real_B_images)
        fake_B_logits = D_B(fake_B_images)
        reconstruction_A_images = G_B2A(fake_B_images)

        fake_A_images = G_B2A(real_B_images)
        fake_A_logits = D_A(fake_A_images)
        reconstruction_B_images = G_A2B(fake_A_images)

        same_B_images = G_A2B(real_B_images)
        same_A_images = G_B2A(real_A_images)

        # Generator Losses
        identity_loss_A = identity_loss_fn(same_A_images, real_A_images) * 5.
        identity_loss_B = identity_loss_fn(same_B_images, real_B_images) * 5.

        G_loss_A2B = GAN_loss_fn(fake_B_logits, real_labels)
        G_loss_B2A = GAN_loss_fn(fake_A_logits, real_labels)
        
        cycle_loss_A = cycle_loss_fn(reconstruction_A_images, real_A_images) * 10.
        cycle_loss_B = cycle_loss_fn(reconstruction_B_images, real_B_images) * 10.

        G_loss = identity_loss_A + identity_loss_B + G_loss_A2B + G_loss_B2A + cycle_loss_A + cycle_loss_B

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

        # Discriminator Losses
        real_A_logits = D_A(real_A_images)
        D_A_loss = (GAN_loss_fn(fake_A_logits, fake_labels) + GAN_loss_fn(real_A_logits, real_labels)) / 2.

        optimizer_D_A.zero_grad()
        D_A_loss.backward()
        optimizer_D_A.step()

        D_B_loss = (GAN_loss_fn(fake_B_logits, fake_labels) + GAN_loss_fn(real_B_logits, real_labels)) / 2.

        optimizer_D_B.zero_grad()
        D_B_loss.backward()
        optimizer_D_B.step()

        print(G_loss, D_A_loss, D_B_loss)