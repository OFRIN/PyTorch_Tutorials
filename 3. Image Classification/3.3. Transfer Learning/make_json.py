import os
import glob
import numpy as np

from utility.utils import write_json

# test
test_names = os.listdir('./dataset/test')

test_dic = {}

for test_name in test_names:
    test_class_name = test_name.split('_')
    test_dic[test_name] = test_class_name[0]

write_json('./data/test.json', test_dic)

# train

train_names = os.listdir('./dataset/train')

train_dic = {}

for train_name in train_names:
    train_class_name = train_name.split('_')
    train_dic[train_name] = train_class_name[0]

write_json('./data/train.json', train_dic)

# validaion

validaion_name = os.listdir('./dataset/validation')

validaion_dic = {}

for validaion_name in validaion_name:
    validaion_class_name = validaion_name.split('_')
    validaion_dic[validaion_name] = validaion_class_name[0]

write_json('./data/validation.json', validaion_dic)
