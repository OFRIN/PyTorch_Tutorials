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

# RGB to BGR
def convert_OpenCV_to_PIL(data):
    return Image.fromarray(data[..., ::-1])

def convert_PIL_to_OpenCV(data):
    return np.asarray(data)

