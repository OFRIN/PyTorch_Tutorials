import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

# weights
conv1 = nn.Conv2d(3, 64, 3, 1, padding=1, bias=False) # [224, 224, 3] -> [224, 224, 64]
conv2 = nn.Conv2d(64, 32, 3, 1, padding=1, bias=False) # [224, 224, 64] -> [224, 224, 32]
conv3 = nn.Conv2d(32, 512, 3, 1, padding=1, bias=False) # [224, 224, 32] -> [224, 224, 512]

# [B, H, W, C] -> [B, C, H, W]
image = np.zeros((1, 3, 224, 224), dtype = np.float32)
image = torch.Tensor(image)

x1 = conv1(image); print(f'{image.size()}->{x1.size()}')
x2 = conv2(x1); print(f'{x1.size()}->{x2.size()}')
x3 = conv3(x2); print(f'{x2.size()}->{x3.size()}')

# Q1. Error
# x4 = conv1(x3)

print(conv1.weight)

