import torch
from torch import nn
from torch.nn import functional as F

# refer : https://pytorch.org/docs/stable/nn.html

############################################################
# Make layers
############################################################
conv = nn.Conv2d(3, 64, 3, 1, padding=1, bias=False) # [224, 224, 3] -> [224, 224, 64]
pool = nn.MaxPool2d((2, 2), 2) # [224, 224, 3] -> [112, 112, 3]

flatten = nn.Flatten() # [224, 224, 3] -> [224 * 224 * 3]
dense1 = nn.Linear(3, stride=2)

relu_fn = nn.ReLU()
softmax_fn = nn.Softmax(dim=1)

############################################################
# Example
############################################################
image_tensor = torch.randn(1, 3, 224, 224)

print(conv(image_tensor).size())
print(pool(image_tensor).size())
print(flatten(image_tensor).size())

