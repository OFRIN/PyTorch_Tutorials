import os
import glob
import numpy as np

from utility.utils import write_json

# test
for domain in ['train', 'validation', 'test']:
    image_names = os.listdir(f'./dataset/{domain}')

    data_dic = {}
    for image_name in image_names:
        class_name = image_name.split('_')
        data_dic[image_name] = class_name[0]

    write_json(f'./data/{domain}.json', data_dic)

# train

# train_names = os.listdir('./dataset/train')

# train_dic = {}

# for train_name in train_names:
#     train_class_name = train_name.split('_')
#     train_dic[train_name] = train_class_name[0]

# write_json('./data/train.json', train_dic)

# # validaion

# validaion_name = os.listdir('./dataset/validation')

# validaion_dic = {}

# for validaion_name in validaion_name:
#     validaion_class_name = validaion_name.split('_')
#     validaion_dic[validaion_name] = validaion_class_name[0]

# write_json('./data/validation.json', validaion_dic)
