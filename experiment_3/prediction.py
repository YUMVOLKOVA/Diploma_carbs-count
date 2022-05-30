import torch
from dataset_class import transformation_test
from train import Model
from skimage import io
import json
import argparse
import os
import sys
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}, {torch.cuda.get_device_name(0)}")
os.chdir('C:\\Users\\Yulia\\Desktop\\carbs-count')


def get_args():
    print("Current arguments:", sys.argv)
    parser = argparse.ArgumentParser()

    parser.add_argument("--runname",
                        help="name this experiment",
                        required=True)
    parser.add_argument("--model",
                        required=True,
                        choices=["resnet50",
                                 "resnet101",
                                 "resnet152",
                                 "densenet121",
                                 "densenet201",
                                 "resnext50_32x4d"])
    parser.add_argument("--weights",
                        required=False,
                        help="continue training (/testing) from these model weights")
    parser.add_argument("--loss",
                        help="what loss function to use",
                        choices=['mae', 'mse', 'rmse'],
                        required=True)
    parser.add_argument("--type_of_training",
                        help="finetuning or fixed feature extractor",
                        choices=['finetuning', 'fixed'],
                        required=True)
    parser.add_argument("--input-file",
                        help="input image",
                        required=False)

    return parser.parse_args()


args = get_args()
model = Model(args)
with open('images/json_depth_test.json') as json_file:
    json_test = json.load(json_file)

if args.input_file:
    images = transformation_test(io.imread(args.input_file))
    images = images.reshape((1, *images.shape))
    truth_for_one = args.input_file.split('\\')[-1]
    with torch.no_grad():
        predict = model.model_dev(images.to(device))
    try:
        print(f'truth: {json_test[str(truth_for_one).split(".")[0]]}, prediction: {predict[0]}')
    except:
        print(f'prediction: {predict[0]}')

else:
    prediction_dict = {}
    for i in tqdm(os.listdir('images/test_depth')):
        images = transformation_test(io.imread('images/test_depth/' + i))
        images = images.reshape((1, *images.shape))
        with torch.no_grad():
            predict = model.model_dev(images.to(device))
            prediction_dict[json_test[i.split('.')[0]]] = float(predict[0])

with open(f'experiment_3/prediction_results/prediction_dict_{args.runname}.json', 'w') as fp:
    json.dump(prediction_dict, fp)