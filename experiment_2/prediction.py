import torch
from dataset_class import transformation_rgb, transformation_depth
from train import Model
from skimage import io
import json
import argparse
import os
import sys
from tqdm import tqdm
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}, {torch.cuda.get_device_name(0)}")
os.chdir('C:\\Users\\Yulia\\Desktop\\carbs-count')

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

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

prediction_dict = {}
prediction_dict_dish = {}
for i in tqdm(os.listdir('images/test_RGB')):
    img_rgb = io.imread('images/test_RGB/' + i)
    img_depth = io.imread('images/test_raw_norm/' + i)
    image_rgb = transformation_rgb(img_rgb)
    image_depth = transformation_depth(img_depth)
    images = torch.cat((transform(image_rgb), transform(image_depth)))
    images = images.reshape((1, *images.shape))
    with torch.no_grad():
        predict = model.model_dev(images.to(device))
        prediction_dict[json_test[i.split('.')[0]]] = float(predict[0])
        prediction_dict_dish[i.split('.')[0]] = {json_test[i.split('.')[0]]: float(predict[0])}
with open(f'experiment_2/prediction_results/prediction_dict_{args.runname}_dish.json', 'w') as fp:
    json.dump(prediction_dict_dish, fp)
with open(f'experiment_2/prediction_results/prediction_dict_{args.runname}.json', 'w') as fp:
    json.dump(prediction_dict, fp)