import os
import cv2
import glob
import numpy as np

from utils import write_json

root_dir = '../../Toy_Dataset/'

train_dic = {}
validation_dic = {}
test_dic = {}

for class_name in os.listdir(root_dir):
    image_dir = root_dir + class_name + '/'
    image_names = os.listdir(image_dir)
        
    length = len(image_names)
    
    train_length = int(length * 0.7)
    validation_length = int(length * 0.1)

    np.random.shuffle(image_names)
    
    train_image_names = image_names[:train_length]; image_names = image_names[train_length:]
    validation_image_names = image_names[:validation_length]; test_image_names = image_names[validation_length:]
    
    # print(f'# {class_name}')
    # print(f'- Train : {len(train_image_names)}')
    # print(f'- Validation : {len(validation_image_names)}')
    # print(f'- Test : {len(test_image_names)}')
    # print()

    for image_name in train_image_names:
        train_dic[image_name] = class_name

    for image_name in validation_image_names:
        validation_dic[image_name] = class_name

    for image_name in test_image_names:
        test_dic[image_name] = class_name

write_json('./data/train.json', train_dic)
write_json('./data/validation.json', validation_dic)
write_json('./data/test.json', test_dic)

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

print(get_class_count(train_dic))
print(get_class_count(validation_dic))
print(get_class_count(test_dic))

# [['cat', 70], ['dog', 70]]
# [['cat', 10], ['dog', 10]]
# [['cat', 20], ['dog', 20]]