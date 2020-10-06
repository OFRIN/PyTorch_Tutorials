import torch
from torch import nn
from torch.nn import functional as F

##################################################
# VGG16
##################################################
conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
pool1 = nn.MaxPool2d((2, 2), stride=2)

conv2_1 = nn.Conv2d(64,128,3,stride=1, padding=1)
conv2_2 = nn.Conv2d(128,128,3,stride=1, padding=1)
pool2 = nn.MaxPool2d((2, 2), stride=2)

conv3_1 = nn.Conv2d(128,256,3,stride=1, padding=1)
conv3_2 = nn.Conv2d(256,256,3,stride=1, padding=1)
conv3_3 = nn.Conv2d(256,256,3,stride=1, padding=1)
pool3 = nn.MaxPool2d((2, 2), stride=2)

conv4_1 = nn.Conv2d(256,512,3,stride=1, padding=1)
conv4_2 = nn.Conv2d(512,512,3,stride=1, padding=1)
conv4_3 = nn.Conv2d(512,512,3,stride=1, padding=1)
pool4 = nn.MaxPool2d((2, 2), stride=2)

conv5_1 = nn.Conv2d(512,512,3,stride=1, padding=1)
conv5_2 = nn.Conv2d(512,512,3,stride=1, padding=1)
conv5_3 = nn.Conv2d(512,512,3,stride=1, padding=1)
pool5 = nn.MaxPool2d((2, 2), stride=2)

Flattens = nn.Flatten()

dense1 = nn.Linear(512 * 7 * 7, 4096)
dense2 = nn.Linear(4096, 4096)
dense3 = nn.Linear(4096, 1000)
relu_fn = nn.ReLU()
softmax_fn = nn.Softmax(dim=1) # [0, 1]

#########################################
# Example
#########################################
tensor = torch.randn(16, 3, 224, 224)

x = relu_fn(conv1_1(tensor))
x = relu_fn(conv1_2(x))
x = pool1(x)

x = relu_fn(conv2_1(x))
x = relu_fn(conv2_2(x))
x = pool2(x)

x = relu_fn(conv3_1(x))
x = relu_fn(conv3_2(x))
x = relu_fn(conv3_3(x))
x = pool3(x)

x = relu_fn(conv4_1(x))
x = relu_fn(conv4_2(x))
x = relu_fn(conv4_3(x))
x = pool4(x)

x = relu_fn(conv5_1(x))
x = relu_fn(conv5_2(x))
x = relu_fn(conv5_3(x))
x = pool5(x)

x = Flattens(x)

x = relu_fn(dense1(x))
x = relu_fn(dense2(x))
logits = dense3(x)
preds = softmax_fn(logits)

print(logits.size(), preds.size())