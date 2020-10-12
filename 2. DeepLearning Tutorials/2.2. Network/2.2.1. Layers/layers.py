import torch
from torch import nn
from torch.nn import functional as F

# refer : https://pytorch.org/docs/stable/nn.html

############################################################
# Make layers
############################################################
conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False) # [224, 224, 3] -> [224, 224, 64]
conv2 = nn.Conv2d(in_channels=64, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # [224, 224, 3] -> [112, 112, 3]

flatten = nn.Flatten() # [224, 224, 3] -> [224 * 224 * 3]
dense1 = nn.Linear(in_features=112 * 112 * 1024, out_features=1)

relu_fn = nn.ReLU()
softmax_fn = nn.Softmax(dim=1)

sigmoid_fn = nn.Sigmoid() # 0 ~ 1
tanh_fn = nn.Tanh() # ~1 ~ 1

############################################################
# Example
############################################################

# batch_size, channels, height, width
image_tensor = torch.randn(4, 3, 224, 224)

x = conv1(image_tensor)
print(x.size()) # 1, 64, 224, 224

x = conv2(x)
print(x.size()) # 1, 1024, 224, 224

x = pool(x)
print(x.size()) # 1, 1024, 112, 112

x = flatten(x)
print(x.size()) # 1, 1024 * 112 * 112

x = dense1(x)
print(x.size())

