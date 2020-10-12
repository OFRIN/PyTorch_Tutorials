import torch
import torch.nn as nn

conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False) # [224, 224, 3] -> [224, 224, 64]
conv2 = nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)

instance_norm1 = nn.InstanceNorm2d(64)
instance_norm2 = nn.InstanceNorm2d(1024)

deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)

block = nn.Sequential(*[conv1, instance_norm1, conv2, instance_norm2])

image_tensor = torch.randn(4, 3, 224, 224)

# y = nn.ReflectionPad2d(1)(image_tensor)
# y = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)(y)
# print(y.size())

# 1.
x = conv1(image_tensor)
x = instance_norm1(x)

x = deconv1(x)
print(x.size())

x = conv2(x)
x = instance_norm2(x)

# 2. 
x = block(image_tensor)

