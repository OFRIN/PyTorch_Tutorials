
import torch

from vgg16 import VGG16

model = VGG16(10)

images = torch.randn(16, 3, 224, 224)
logits, preds = model.forward(images)

print(images.size())
print(preds.size())

