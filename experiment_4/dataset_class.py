from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from skimage import io
import json
import os
import torch

def transform_data(element):
    return np.array([element], dtype=np.float32)

transformation = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class FoodDataset(Dataset):
    def __init__(self, carbs_file, image_dir, type_data):
        self.image_dir = image_dir
        splitted_path = self.image_dir.split('/')[-1]
        splitted_path = splitted_path.split('_')[0]
        if type_data == 'train':
            self.depth_dir = 'images/' + str(splitted_path) + '_depth_aug'
        elif type_data == 'test':
            self.depth_dir = 'images/' + str(splitted_path) + '_depth'
        self.transform_rgb = transformation
        self.transform_depth = transformation

        with open(carbs_file) as json_file:
            self.data = json.load(json_file)
        self.carbs_values_tuples = tuple(self.data.items())
        self.transform_ = transforms.Compose([transforms.ToPILImage(),
                                             transforms.ToTensor()])

    def __len__(self):
        return len(list(self.data.items()))

    def __getitem__(self, idx):
        element = self.carbs_values_tuples[idx]
        img_name = os.path.join(self.image_dir, element[0] + '.png')
        image_rgb = io.imread(img_name)
        image_rgb = self.transform_rgb(image_rgb)
        depth_name = os.path.join(self.depth_dir, element[0] + '.jpeg')
        image_depth = io.imread(depth_name)
        image_depth = self.transform_depth(image_depth)

        image = torch.cat((self.transform_(image_rgb), self.transform_(image_depth)))
        sample = {"fname": element[0], "image": image}

        value = transform_data(element[1])
        sample['carbs'] = value
        return sample