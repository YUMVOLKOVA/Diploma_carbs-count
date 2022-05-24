from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
import numpy as np
from skimage import io
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch


def transform_data(element):
    return np.array([element], dtype=np.float32)


food_image_transform = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((224, 224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class FoodDataset(Dataset):
    def __init__(self, *, carbs_file, image_dir, depth=False):
        # dir with rgb photo
        self.image_dir = image_dir
        if depth:
            splitted_path = self.image_dir.split('\\')

            self.depth_dir = "C:\\Users\\Yulia\\Desktop\\Carbohydrate-counting\\images\\" + splitted_path[-1].split('/')[1].split('_')[0] + '_raw'

        self.transform = food_image_transform
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
        if self.depth_dir:
            depth_name = os.path.join(self.depth_dir, element[0] + '.png')

            image_depth = io.imread(depth_name)
            image = torch.cat((self.transform_(image_rgb), self.transform_(image_depth.astype(np.uint8))))
            image = image.reshape(3, 800, -1)

            sample = {"fname": element[0], "image": image}
        else:

            sample = {"fname": element[0], "image": image_rgb}

        value = transform_data(element[1])
        sample['carbs'] = value
        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample
