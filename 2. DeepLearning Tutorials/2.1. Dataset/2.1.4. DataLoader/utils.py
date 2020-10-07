import json
import numpy as np

def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def get_one_hot_vector(index, classes):
    vector = np.zeros((classes), dtype = np.float32)
    vector[index] = 1.
    return vector