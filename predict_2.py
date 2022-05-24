import torch
from dataset_class import food_image_transform
from train import get_args, Model
from skimage import io
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}, {torch.cuda.get_device_name(0)}")

args = get_args(train=False)
model = Model(args)
images = food_image_transform(io.imread(args.input_file))
# нужно чтение для целой папки
print(images.shape)

images = images.reshape((1, *images.shape))
print(images.shape)
truth_for_one = args.input_file.split('\\')[-1]
print(truth_for_one)
json_test_path = "C:\\Users\\Yulia\\Desktop\\Carbohydrate-counting\\images\\json_depth_test.json"
with open(json_test_path) as json_file:
    json_test = json.load(json_file)

with torch.no_grad():
    predict = model.model_dev(images.to(device))
try:
    print(f'truth: {json_test[str(truth_for_one).split(".")[0]]}, prediction: {predict[0]}')
except:
    print(f'prediction: {predict[0]}')