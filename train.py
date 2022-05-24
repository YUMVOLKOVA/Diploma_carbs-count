import argparse
import datetime
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_class import FoodDataset
from models.pretrained import PretrainedModel

params = {'num_epochs': 40,
          'batch_size': 128,
          'num_output_neurons': 1,
          'lr': 1e-3,
          'image_dir_train': "images/train_RGB",
          'image_dir_test': "images/test_RGB",
          'logdir': 'runs/',
          'carbs_file_train': "images/json_depth_train.json",
          'carbs_file_test': "images/json_depth_test.json"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}, {torch.cuda.get_device_name(0)}")


def get_args(train=True):
    print("Current arguments:", sys.argv)
    parser = argparse.ArgumentParser()

    if train:
        parser.add_argument("--runname",
                            help="name this experiment",
                            required=True)
    else:
        parser.add_argument("--input-file",
                            help="input image",
                            required=True)
        parser.add_argument("--runname",
                            help="name this experiment",
                            required=False)

    # добавить тут еще моделей
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
    parser.add_argument("--depth",
                        required=False,
                        help="if you need depth maps")

    return parser.parse_args()


class Model(nn.Module):
    def __init__(self, arguments, batch_size=64, num_output_neurons=1, logdir='runs/'):
        super().__init__()
        self.args = arguments
        self.batch_size = batch_size
        self.num_output_neurons = num_output_neurons
        self.model = PretrainedModel(self.num_output_neurons,
                                     pytorch_model=self.args.model)
        self.logdir = (logdir + datetime.datetime.now().replace(microsecond=0).isoformat().replace(":",
                                                                                                   ".") + "-" + self.args.runname + '-' + self.model.name)
        self.writer = SummaryWriter(self.logdir)
        print(f"tensorboard logdir: {self.writer.log_dir}")
        if self.args.weights:
            print(f"Loading model weights from {self.args.weights}")
            self.model.load(self.args.weights)
        print("model:", self.model.name)
        self.model_dev = self.model.to(device)
        print(self.model_dev)

    def calculate_loss(self, y, y_hat):
        loss = nn.SmoothL1Loss()
        output = loss(y_hat, y)
        return output

    def criterion_rel_error(self, pred, truth):
        ret = torch.abs(1 - pred / truth)
        ret[torch.isnan(ret)] = 0  # if truth = 0 relative error is undefined
        return torch.mean(ret)

    def count_parameters(self, model_):
        trainable_params = sum(p.numel() for p in model_.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model_.parameters())
        return trainable_params, total_params

    def for_test(self, test_data):
        print('start with test ')
        loss = []
        relative = []
        with torch.no_grad():
            self.model.eval()
            for data in tqdm(test_data):
                batch = data['image'].to(device)
                target = data['carbs'].to(device)
                outputs = self.model_dev(batch)
                loss.append(float(self.calculate_loss(target, outputs)))
                relative.append(float(self.criterion_rel_error(outputs, target)))
        print('done with current test counting ')
        return loss, relative

    def fit(self, num_epochs, train_data, test_data, lr):
        optimizer = Adam(self.model_dev.parameters(), lr=lr)
        trainable_params, total_params = self.count_parameters(self.model_dev)
        print(f"Parameters: {trainable_params} trainable, {total_params} total")

        batch_idx = -1

        test_loss, relative = self.for_test(test_data)
        self.writer.add_scalar('MAE test for epoch', np.mean(test_loss), 0)
        self.writer.add_scalar('std MAE test for epoch', np.std(test_loss), 0)
        self.writer.add_scalar('relative loss test for epoch', np.mean(relative), 0)
        print('starting my epochs')

        for epoch in tqdm(range(1, num_epochs + 1)):
            print(f'epoch: {epoch}')
            train_loss_for_epoch = []
            relat_for_batch = []

            for epoch_batch_idx, data in enumerate(tqdm(train_data), 0):
                batch_idx += 1
                batch = data['image'].to(device)
                target = data['carbs'].to(device)
                print(target.shape)

                optimizer.zero_grad()
                outputs = self.model_dev(batch)
                print(f'output shape: {outputs.shape}')
                loss = self.calculate_loss(target, outputs)
                relat = self.criterion_rel_error(outputs, target)
                print(loss)

                loss.backward()
                optimizer.step()

                train_loss_for_epoch.append(float(loss))
                relat_for_batch.append(float(relat))
                self.writer.add_scalar('train loss by batch', float(loss), batch_idx)
                self.writer.add_scalar('train loss by batch relative', float(relat), batch_idx)

            test_loss, relative = self.for_test(test_data)
            self.writer.add_scalar('MAE test for epoch', np.mean(test_loss), epoch)
            self.writer.add_scalar('std MAE test for epoch', np.std(test_loss), epoch)
            self.writer.add_scalar('relative loss test for epoch', np.mean(relative), epoch)
            print(f'mean test loss for epoch {epoch} is {np.mean(test_loss)}')

            self.writer.add_scalar('MAE train for epoch', np.mean(train_loss_for_epoch), epoch)
            self.writer.add_scalar('std MAE train for epoch', np.std(train_loss_for_epoch), epoch)
            self.writer.add_scalar('relative loss train for epoch', np.mean(relat_for_batch), epoch)
            print(f'mean train loss for epoch {epoch} is {np.mean(train_loss_for_epoch)}')

            self.model.save(self.model_dev, f'epoch_{epoch}', self.logdir)
            print(f'done with model save for epoch {epoch}')

        self.writer.close()
        print('done fitting epochs')


if __name__ == "__main__":
    args = get_args(train=True)
    model = Model(args,
                  batch_size=params['batch_size'],
                  num_output_neurons=params['num_output_neurons'],
                  logdir=params['logdir'])
    print('done with init model')

    train = FoodDataset(carbs_file=params['carbs_file_train'],
                        image_dir=params['image_dir_train'],
                        depth=args.depth)
    train = DataLoader(train,
                       batch_size=params['batch_size'],
                       shuffle=True,
                       pin_memory=True)
    print('done with loading train data')

    test = FoodDataset(carbs_file=params['carbs_file_test'],
                       image_dir=params['image_dir_test'],
                       depth=args.depth)
    test = DataLoader(test,
                      batch_size=params['batch_size'],
                      shuffle=True,
                      pin_memory=True)
    print('done with loading test data')

    model.fit(num_epochs=params['num_epochs'],
              train_data=train,
              test_data=test,
              lr=params['lr'])
