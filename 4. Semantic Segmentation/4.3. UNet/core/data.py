
'''
import cv2
import numpy as np
import tiffile as tiff

images = tiff.imread('C:/Class/DATA/ISBI/train-volume.tif')
labels = tiff.imread('C:/Class/DATA/ISBI/train-labels.tif')
print(images.shape)

for i in range(len(images)):
    # print(i.shape)

    print(np.min(labels[i]))
    print(np.max(labels[i]))

    # cv2.imshow('image', images[i])
    # cv2.imshow('label', labels[i])
    # cv2.waitKey(0)
'''
import cv2
import numpy as np
import os
import tiffile as tiff

from torch.utils.data import Dataset

from utility.utils import convert_OpenCV_to_PIL, transpose

class ISBI_Dataset_For_Training(Dataset):
    def __init__(self,image_dir,transforms, class_names):
        self.image_dir = image_dir
        self.transforms = transforms

        self.images = tiff.imread(os.path.join(image_dir,'train-volume.tif'))
        self.labels = tiff.imread(os.path.join(image_dir,'train-labels.tif'))
        
        self.class_names = class_names
        self.classes = len(class_names)
        self.class_dic = {name:label for label, name in enumerate(self.class_names)}

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        train_img = self.images[index] # [256, 256, 1]
        train_roi = self.labels[index] # 0 or 255

        train_img = self.transforms(train_img) # 0~1
        train_roi = self.transforms(train_roi) # 0~1

        return train_img, train_roi

class ISBI_Dataset_For_Testing(Dataset):
    def __init__(self, image_path, transforms, class_names):
        self.transforms = transforms

        self.images = tiff.imread(image_path)
        
        self.class_names = class_names
        self.classes = len(class_names)
        self.class_dic = {name:label for label, name in enumerate(self.class_names)}

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = self.images[index] # [256, 256, 1]
        image = self.transforms(image) # 0~1
        return image

if __name__ == '__main__':
    from torchvision import transforms

    image_dir = 'C:/Class/DATA/ISBI/'
    class_names = ['cell']
    test_transforms = transforms.Compose(
        [
            convert_OpenCV_to_PIL,
            transforms.Resize(256),
            transforms.ToTensor(), # / 255
        ]
    )

    dataset = ISBI_Dataset(image_dir,test_transforms,class_names)

    for index in range(len(dataset)):
        image, label = dataset[index]
        print(image.shape, label.shape)

        image = image.numpy()*255
        label = label.numpy()*255

        # C, H, W -> H, W, C
        # 0, 1, 2 -> 1, 2, 0
        image = transpose(image, (1, 2, 0))
        label = transpose(label, (1, 2, 0))

        image = image.astype(np.uint8)
        label = label.astype(np.uint8)

        cv2.imshow('image',image)
        cv2.imshow('label',label)
        cv2.waitKey(0)