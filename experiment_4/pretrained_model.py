import torch
import torch.nn as nn
from torchvision import models

class PretrainedModel(nn.Module):
    def __init__(self, pytorch_model, type_of_training):
        super(PretrainedModel, self).__init__()
        self.name = f'{pytorch_model}'
        self.pytorch_model = pytorch_model
        self.type_of_training = type_of_training

        self.model = getattr(models, self.pytorch_model)(pretrained=True)
        weight = self.model.conv1.weight.clone()
        self.model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.model.conv1.weight[:, :3] = weight
            self.model.conv1.weight[:, 3:] = weight

        if self.type_of_training == 'fixed':
            count = 1
            for param in self.model.parameters():
                if count == 1:
                    param.requires_grad = True
                    count += 1
                else:
                    param.requires_grad = False

        llayer = self.get_last_layer()
        num_ftrs = getattr(self.model, llayer).in_features
        setattr(self.model, llayer, nn.Linear(num_ftrs, 1))

    def get_last_layer(self):
        if self.pytorch_model.startswith('resnet') or self.pytorch_model.startswith('resnext'):
            return 'fc'
        if self.pytorch_model.startswith('densenet'):
            return 'classifier'
        raise Exception('what is this model')

    def save(self, model, run_name, path):
        full_path = path + "/" + run_name + ".pt"
        torch.save(model.state_dict(), full_path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        return self.model(x)