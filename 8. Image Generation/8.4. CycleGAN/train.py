
import cv2
import torch
import itertools

from PIL import Image
from torchvision import transforms

from core.dataset import ImageDataset
from core.networks import Generator, Discriminator
from core.utils import *

from util.time_utils import Timer
from util.utils import *

# 1. Define
# root_dir = './datasets/horse2zebra/'
root_dir = 'C:/DB/CycleGAN/horse2zebra/horse2zebra/'

learning_rate = 0.0002
batch_size = 1
image_size = 256
A_channels = 3
B_channels = 3
max_epoch = 200

# 2. Dataset
transform = transforms.Compose(
    [
        transforms.Resize(int(image_size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),

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

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(max_epoch, 0, max_epoch // 2).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(max_epoch, 0, max_epoch // 2).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(max_epoch, 0, max_epoch // 2).step)

# 6. Training
# input_A = torch.Tensor(batch_size, A_channels, image_size, image_size)
# input_B = torch.Tensor(batch_size, B_channels, image_size, image_size)
real_labels = torch.autograd.Variable(torch.Tensor(batch_size, 1).fill_(1.0), requires_grad=False).cuda()
fake_labels = torch.autograd.Variable(torch.Tensor(batch_size, 1).fill_(0.0), requires_grad=False).cuda()

train_timer = Timer()
train_avg = Average_Meter([
    'identity_A_loss', 'identity_B_loss', 
    'cycle_A_loss', 'cycle_B_loss', 
    'G_A2B_loss', 'G_B2A_loss', 
    'G_loss',
    'D_A_loss', 'D_B_loss',
])

for epoch in range(1, max_epoch + 1):

    train_avg.clear()
    train_timer.tik()

    for real_A_images, real_B_images in zip(A_loader, B_loader):
        real_A_images = real_A_images.cuda()
        real_B_images = real_B_images.cuda()
        
        # 1. Identity Loss
        same_B_images = G_A2B(real_B_images)
        same_A_images = G_B2A(real_A_images)
        
        identity_A_loss = identity_loss_fn(same_A_images, real_A_images) * 5.
        identity_B_loss = identity_loss_fn(same_B_images, real_B_images) * 5.

        # 2. GAN Loss
        fake_B_images = G_A2B(real_A_images)
        fake_B_logits = D_B(fake_B_images)

        fake_A_images = G_B2A(real_B_images)
        fake_A_logits = D_A(fake_A_images)

        G_A2B_loss = GAN_loss_fn(fake_B_logits, real_labels)
        G_B2A_loss = GAN_loss_fn(fake_A_logits, real_labels)
        
        # 3. Cycle Consistency Loss
        reconstruction_A_images = G_B2A(fake_B_images)
        reconstruction_B_images = G_A2B(fake_A_images)

        cycle_A_loss = cycle_loss_fn(reconstruction_A_images, real_A_images) * 10.
        cycle_B_loss = cycle_loss_fn(reconstruction_B_images, real_B_images) * 10.

        G_loss = identity_A_loss + identity_B_loss + G_A2B_loss + G_B2A_loss + cycle_A_loss + cycle_B_loss

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

        # Discriminator Losses
        real_A_logits = D_A(real_A_images)
        fake_A_logits = D_A(fake_A_images.detach())

        D_A_loss = (GAN_loss_fn(fake_A_logits, fake_labels) + GAN_loss_fn(real_A_logits, real_labels)) / 2.

        optimizer_D_A.zero_grad()
        D_A_loss.backward(retain_graph=True)
        optimizer_D_A.step()

        real_B_logits = D_B(real_B_images)
        fake_B_logits = D_B(fake_B_images.detach())

        D_B_loss = (GAN_loss_fn(fake_B_logits, fake_labels) + GAN_loss_fn(real_B_logits, real_labels)) / 2.

        optimizer_D_B.zero_grad()
        D_B_loss.backward(retain_graph=True)
        optimizer_D_B.step()

        train_avg.add({
            'identity_A_loss' : identity_A_loss.item(), 'identity_B_loss' : identity_B_loss.item(), 
            'cycle_A_loss' : cycle_A_loss.item(), 'cycle_B_loss' : cycle_B_loss.item(), 
            'G_A2B_loss' : G_A2B_loss.item(), 'G_B2A_loss' : G_B2A_loss.item(), 
            'G_loss' : G_loss.item(),
            'D_A_loss' : D_A_loss.item(), 'D_B_loss' : D_B_loss.item(),
        })

    data = [epoch] + train_avg.get(clear=True) + [train_timer.tok()]
    print('[i] epoch={}, identity_A_loss={:.4f}, identity_B_loss={:.4f}, cycle_A_loss={:.4f}, cycle_B_loss={:.4f}, G_A2B_loss={:.4f}, G_B2A_loss={:.4f}, G_loss={:.4f}, D_A_loss={:.4f}, D_B_loss={:.4f}, time={}sec'.format(*data))

    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    torch.save(G_A2B.state_dict(), './model/G_A2B.pth')
    torch.save(G_B2A.state_dict(), './model/G_B2A.pth')
    torch.save(D_A.state_dict(), './model/D_A.pth')
    torch.save(D_B.state_dict(), './model/D_B.pth')