from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from skimage import io
import json
import os


def transform_data(element):
    return np.array([element], dtype=np.float32)


transformation_train = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((224, 224)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transformation_test = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class FoodDataset(Dataset):
    def __init__(self, carbs_file, image_dir, type_data):
        self.image_dir = image_dir
        self.type_data = type_data
        if self.type_data == 'train':
            self.transform = transformation_train
        elif self.type_data == 'test':
            self.transform = transformation_test
        with open(carbs_file) as json_file:
            self.data = json.load(json_file)
        self.carbs_values_tuples = tuple(self.data.items())

    def __len__(self):
        return len(list(self.data.items()))

    def __getitem__(self, idx):
        element = self.carbs_values_tuples[idx]
        img_name = os.path.join(self.image_dir, element[0] + '.jpeg')
        image_rgb = io.imread(img_name)
        sample = {"fname": element[0], "image": image_rgb}
        value = transform_data(element[1])
        sample['carbs'] = value
        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample