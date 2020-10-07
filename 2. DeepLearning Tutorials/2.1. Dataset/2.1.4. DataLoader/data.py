
import cv2
import numpy as np

from torch.utils.data import Dataset

from utils import read_json, get_one_hot_vector

class Single_Classification_Dataset(Dataset):
    def __init__(self, image_dir, transforms, json_path, class_names):
        self.image_dir = image_dir
        self.data_dic = read_json(json_path)
        
        self.transforms = transforms

        self.class_names = class_names
        self.classes = len(class_names)
        self.class_dic = {name : label for label, name in enumerate(self.class_names)}

        self.image_names = list(self.data_dic.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        class_name = self.data_dic[image_name]

        image_path = self.image_dir + class_name + '/' + image_name
        class_index = self.class_dic[class_name]

        image = cv2.imread(image_path)
        image = self.transforms(image)

        label = get_one_hot_vector(class_index, self.classes)
        
        return image, label