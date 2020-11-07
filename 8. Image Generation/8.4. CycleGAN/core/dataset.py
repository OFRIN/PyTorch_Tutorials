

import cv2
import glob
import torch

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, mode='trainA'):
        self.transform = transform
        self.image_paths = glob.glob(root_dir + f'{mode}/*')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return image

if __name__ == '__main__':
    root_dir = 'C:/DB/CycleGAN/horse2zebra/horse2zebra/'

    A_dataset = ImageDataset(root_dir, mode='trainA')
    B_dataset = ImageDataset(root_dir, mode='trainB')

    for i in range(len(A_dataset)):
        cv2.imshow('A', A_dataset[i])
        cv2.imshow('B', B_dataset[i])
        cv2.waitKey(0)
