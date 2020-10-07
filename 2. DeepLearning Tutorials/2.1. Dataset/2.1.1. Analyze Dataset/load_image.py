import os
import cv2
import glob

root_dir = '../../Toy_Dataset/'

for class_name in os.listdir(root_dir):
    for image_path in glob.glob(root_dir + class_name + '/*'):
        
        image = cv2.imread(image_path)

        cv2.imshow(class_name, image)
        cv2.waitKey(0)