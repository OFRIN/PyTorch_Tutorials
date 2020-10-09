import os
import cv2
import glob

from opencv_transforms import transforms

random_crop_fn = transforms.RandomResizedCrop(224)
random_hflip_fn = transforms.RandomHorizontalFlip()

root_dir = '../../Toy_Dataset/'

for class_name in os.listdir(root_dir):
    for image_path in glob.glob(root_dir + class_name + '/*'):
        
        image = cv2.imread(image_path)

        crop_image = random_crop_fn(image)
        hflip_image = random_hflip_fn(image)

        cv2.imshow('original', image)
        cv2.imshow('crop', crop_image)
        cv2.imshow('flip', hflip_image)
        cv2.waitKey(0)
