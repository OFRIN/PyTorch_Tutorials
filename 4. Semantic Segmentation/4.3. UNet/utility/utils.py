import json
import numpy as np

from PIL import Image

def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def get_one_hot_vector(index, classes):
    vector = np.zeros((classes), dtype = np.float32)
    vector[index] = 1.
    return vector

def csv_print(data_list, log_path = './log.csv'):
    string = ''
    for data in data_list:
        if type(data) != type(str):
            data = str(data)
        string += (data + ',')
    
    if log_path is not None:
        f = open(log_path, 'a+')
        f.write(string + '\n')
        f.close()

# RGB to BGR
def convert_OpenCV_to_PIL(data):
    return Image.fromarray(data)

def convert_PIL_to_OpenCV(data):
    return np.asarray(data)

def transpose(data, condition):
    return np.transpose(data, condition)

def write_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent = '\t')
