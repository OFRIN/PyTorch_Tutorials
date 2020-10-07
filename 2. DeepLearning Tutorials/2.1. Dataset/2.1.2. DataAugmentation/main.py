
import os
import cv2
import glob

from augmentation.augment_utils import Random_HorizontalFlip
from augmentation.augment_utils import Random_ColorJitter

root_dir = '../../Toy_Dataset/'

func1 = Random_HorizontalFlip(0.5)
func2 = Random_ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1)

for class_name in os.listdir(root_dir):
    for image_path in glob.glob(root_dir + class_name + '/*'):
        
        image = cv2.imread(image_path)

        flip_image = func1(image)
        color_image = func2(image)

        cv2.imshow('original', image)
        cv2.imshow('flip', flip_image)
        cv2.imshow('color', color_image)
        cv2.waitKey(0)
