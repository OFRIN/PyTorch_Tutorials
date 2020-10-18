import os
import cv2
import glob
import json
import numpy as np

def write_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent = '\t')

root_dir = './dataset/'
data_dic = {
    'train' : {},
    'validation' : {},
    'test' : {}
}

for dataset in ['validation', 'train', 'test']:

    for classes in ['NORMAL', 'PNEUMONIA']:
        data_folders = glob.glob(root_dir+f'{dataset}/{classes}/*')

        for data_folder in data_folders:
            data_folder = data_folder.replace(root_dir+f'{dataset}/', '')
            data_folder = data_folder.replace('\\', '/')

            data_dic[dataset][data_folder] = classes

write_json('./train.json', data_dic['train'])
write_json('./validation.json', data_dic['validation'])
write_json('./test.json', data_dic['test'])

#########################################################################################################################3
# Debug
#########################################################################################################################3
def get_class_count(dictionary):
    class_dic = {}

    for image_name in list(dictionary.keys()):
        try:
            class_dic[dictionary[image_name]] += 1
        except KeyError:
            class_dic[dictionary[image_name]] = 1

    class_names = list(class_dic.keys())
    return [[class_name, class_dic[class_name]] for class_name in sorted(class_names)]

print(get_class_count(data_dic['train']))
print(get_class_count(data_dic['validation']))
print(get_class_count(data_dic['test']))

# [['NORMAL', 1341], ['PNEUMONIA', 3875]]
# [['NORMAL', 8], ['PNEUMONIA', 8]]
# [['NORMAL', 234], ['PNEUMONIA', 390]]

