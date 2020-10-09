
from torchvision import models
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

mobilenet_v2_model = models.mobilenet_v2(pretrained=True)

