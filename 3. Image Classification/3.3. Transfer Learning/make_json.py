import os
import glob
import numpy as np

from utility.utils import write_json

for domain in ['train', 'validation', 'test']:
    image_names = os.listdir(f'./dataset/{domain}')

    data_dic = {}
    for image_name in image_names:
        class_name = image_name.split('_')
        data_dic[image_name] = class_name[0]

    write_json(f'./data/{domain}.json', data_dic)

