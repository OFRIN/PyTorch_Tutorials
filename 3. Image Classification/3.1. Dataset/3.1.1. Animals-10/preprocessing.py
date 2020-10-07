
# OpenCV
import cv2

# image = cv2.imread('./dataset/raw-img/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg')
# print(image.shape)

# cv2.imshow('show', image)
# cv2.waitKey(0)

# os
import os

# image_dir = './dataset/raw-img/cane/'
# image_names = os.listdir(image_dir)

# for image_name in image_names:
#     print(image_name)

# Q
label_dic = {
    "cane": "dog", 
    "cavallo": "horse", 
    "elefante": "elephant", 
    "farfalla": "butterfly", 
    "gallina": "chicken", 
    "gatto": "cat", 
    "mucca": "cow", 
    "pecora": "sheep", 
    "scoiattolo": "squirrel",
    "ragno": "spider",
}
# print(label_dic['cane'])

#############################################################
# 
#############################################################
# # "image_name, label"
# cane_dir = './dataset/raw-img/cane/'
# cavallo_dir = './dataset/raw-img/cavallo/'
# elefante_dir = './dataset/raw-img/elefante/'
# farfalla_dir = './dataset/raw-img/farfalla/'
# gallina_dir = './dataset/raw-img/gallina/'
# gatto_dir = './dataset/raw-img/gatto/'
# mucca_dir = './dataset/raw-img/mucca/'
# pecora_dir = './dataset/raw-img/pecora/'
# ragno_dir = './dataset/raw-img/ragno/'
# scoiattolo_dir = './dataset/raw-img/scoiattolo/'


# image_names = os.listdir(cane_dir)

# for i in range(0,len(image_names)):

#     img_names = image_names[i]
#     # os.path.split(cane_dir)
# # 
#     dir_split = os.path.split(cane_dir)
#     names = dir_split[0]
#     fn = os.path.basename(names) 

#     print(img_names,fn)

# #     label_dic[fn]

#############################################################
# 
#############################################################
import shutil
import numpy as np

root_dir = './dataset/raw-img/'

train_dir = './dataset/train/'
validation_dir = './dataset/validation/'
test_dir = './dataset/test/'

class_names = os.listdir(root_dir)

for class_name in class_names:
    image_dir = root_dir + class_name + '/'
    image_names = os.listdir(image_dir)

    label = label_dic[class_name]

    length = len(image_names)
    
    train_length = int(length * 0.7) # 70%
    validation_length = int(length * 0.1) # 10%
    test_length = length - train_length - validation_length

    np.random.shuffle(image_names)

    # 1. 
    # train_image_names = image_names[:train_length]
    # validation_image_names = image_names[train_length:train_length+validation_length]
    # test_image_names = image_names[train_length+validation_length:]

    # 2. 
    train_image_names = image_names[:train_length]; image_names = image_names[train_length:]
    validation_image_names = image_names[:validation_length]
    test_image_names = image_names[validation_length:]
    
    # print(image_dir, label, len(train_image_names), len(validation_image_names), len(test_image_names))

    for image_name in train_image_names:
        src_image_path = image_dir + image_name
        dst_image_path = train_dir + f'{label}_{image_name}'

        shutil.copy(src_image_path, dst_image_path)
    
    for image_name in validation_image_names:
        src_image_path = image_dir + image_name
        dst_image_path = validation_dir + f'{label}_{image_name}'
        
        shutil.copy(src_image_path, dst_image_path)

    for image_name in test_image_names:
        src_image_path = image_dir + image_name
        dst_image_path = test_dir + f'{label}_{image_name}'
        
        shutil.copy(src_image_path, dst_image_path)
    