# mean baseline: always predicts the value that is the average value in the dataset 

from pathlib import Path
import json
import numpy as np
import argparse
import random
import torch
import torch.nn as nn


# parser = argparse.ArgumentParser()
# parser.add_argument("--runname", help="name this experiment", required=True)
# parser.add_argument("--carbs_file_train",
#                     help="input data dir",
#                     required=True)
# parser.add_argument("--carbs_file_test",
#                     help="input data dir",
#                     required=True)
# parser.add_argument("--image_dir_train",
#                     help="input data dir",
#                     required=True)
# parser.add_argument("--image_dir_test",
#                     help="input data dir",
#                     required=True)
# args = parser.parse_args()

# carbs_file_train = args.carbs_file_train
# carbs_file_test = args.carbs_file_test
# image_dir_train = args.image_dir_train
# image_dir_test = args.image_dir_test

def baseline_model(values):
    mean = np.mean(values)

    def model():
        return mean

    return model

def calculate_loss(y, y_hat):
    loss = []
    for i in y:
        loss.append(i-y_hat)
    output = np.mean(np.abs(loss))       

    return output


carbs_file_train = "C:\\Users\\Yulia\\Desktop\\Carbohydrate-counting\\working_with_dataset\\json_train.json"
carbs_file_test = "C:\\Users\\Yulia\\Desktop\\Carbohydrate-counting\\working_with_dataset\\json_test.json"
image_dir_train = "C:\\Users\\Yulia\\Desktop\\Carbohydrate-counting\\images\\train"
image_dir_test = "C:\\Users\\Yulia\\Desktop\\Carbohydrate-counting\\images\\test"


with open(carbs_file_train) as f:
    train = json.load(f)
    
with open(carbs_file_test) as f:
    test = json.load(f)

train_array = [train[i] for i in train]
train_array_mean = np.mean(train_array)
print(f'train_array_mean: {train_array_mean}')


train_array_std = np.std(train_array)
print(f'train_array_std: {train_array_std}')

test_array = [test[i] for i in test]
test_array_mean = np.mean(test_array)
print(f'test_array_mean: {test_array_mean}')

test_array_std = np.std(test_array)
print(f'test_array_std: {test_array_std}')


baseline = baseline_model(train_array)
print(f'baseline: {baseline()}')

loss_train = calculate_loss(train_array, baseline())
loss_test = calculate_loss(test_array, baseline())

print(f'''baseline loss for train: {loss_train},
baseline loss for test: {loss_test}''')

