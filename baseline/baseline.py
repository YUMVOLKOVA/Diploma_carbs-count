"""
mean baseline: always predicts the value that is the average value in the dataset
"""

import argparse
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
import os



parser = argparse.ArgumentParser()

parser.add_argument("--wdir",
                    help="working",
                    required=True)
parser.add_argument("--images_train",
                    help="train image dir",
                    required=True)
parser.add_argument("--images_test",
                    help="test image dir",
                    required=True)
parser.add_argument("--target_train",
                    help="train target dir",
                    required=True)
parser.add_argument("--target_test",
                    help="test target dir",
                    required=True)
parser.add_argument("--loss",
                    help="what loss function to use",
                    choices=['mae', 'mse', 'msle'],
                    required=True)
args = parser.parse_args()


def baseline_model(values):
    mean = np.mean(values)

    def model():
        return mean

    return model


def calculate_loss(y, y_hat):
    if args.loss == 'mae':
        output = mean_absolute_error(y, y_hat)
    elif args.loss == 'mse':
        output = mean_squared_error(y, y_hat)
    elif args.loss == 'msle':
        output = mean_squared_log_error(y, y_hat)

    return output


if __name__ == "__main__":
    os.chdir(args.wdir)
    carbs_file_train = args.target_train
    with open(carbs_file_train) as f:
        train = json.load(f)

    carbs_file_test = args.target_test
    with open(carbs_file_test) as f:
        test = json.load(f)

    image_dir_train = args.images_train
    image_dir_test = args.images_test

    train_array = [train[i] for i in train]
    test_array = [test[i] for i in test]

    baseline = baseline_model(train_array)
    print(f'baseline: {baseline()}')
    loss_train = calculate_loss(train_array, [baseline()] * len(train_array))
    loss_test = calculate_loss(test_array, [baseline()] * len(test_array))

    print(f'''
    baseline {args.loss} loss for train: {loss_train},
    baseline {args.loss} loss for test: {loss_test}
    ''')
