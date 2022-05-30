import argparse
import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import DataLoader

from pretrained_model import PretrainedModel
from dataset_class import FoodDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}, {torch.cuda.get_device_name(0)}")

os.chdir('C:\\Users\\Yulia\\Desktop\\carbs-count')

params = {'num_epochs': 40,
          'batch_size': 64,
          'lr': 1e-3}


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
                        choices=['mae', 'mse', 'msle'],
                        required=True)
    parser.add_argument("--type_of_training",
                        help="finetuning or fixed feature extractor",
                        choices=['finetuning', 'fixed'],
                        required=True)
    return parser.parse_args()


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


class Model(nn.Module):
    def __init__(self, arguments, batch_size=64, logdir='experiment_1/runs/'):
        super().__init__()
        self.args = arguments
        self.batch_size = batch_size
        self.model = PretrainedModel(pytorch_model=self.args.model, type_of_training=self.args.type_of_training)
        self.logdir = (logdir + self.args.runname + '-' + self.model.name)
        self.writer = SummaryWriter(self.logdir)
        print(f"tensorboard logdir: {self.writer.log_dir}")
        if self.args.weights:
            print(f"Loading model weights from {self.args.weights}")
            self.model.load(self.args.weights)
        print("model:", self.model.name)
        self.model_dev = self.model.to(device)
        print(self.model_dev)

    def loss_mae(self, y, y_hat):
        loss = nn.L1Loss()
        output = loss(y_hat, y)
        return output

    def loss_mse(self, y, y_hat):
        loss = nn.MSELoss()
        output = loss(y_hat, y)
        return output

    def loss_msle(self, y, y_hat):
        loss = RMSLELoss()
        output = loss(y_hat, y)
        return output

    def calculate_loss(self, y, y_hat):
        if self.args.loss == 'mae':
            output = self.loss_mae(y, y_hat)
        elif self.args.loss == 'mse':
            output = self.loss_mse(y, y_hat)
        elif self.args.loss == 'msle':
            output = self.loss_msle(y, y_hat)
        return output

    def count_parameters(self, model_):
        trainable_params = sum(p.numel() for p in model_.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model_.parameters())
        return trainable_params, total_params

    def for_test(self, test_data):
        self.model_dev.eval()
        with torch.no_grad():
            loss = []
            loss_mae = []
            loss_mse = []
            loss_msle = []
            for data in tqdm(test_data):
                batch = data['image'].to(device)
                target = data['carbs'].to(device)
                outputs = self.model_dev(batch)
                loss.append(float(self.calculate_loss(target, outputs)))
                loss_mae.append(float(self.loss_mae(target, outputs)))
                loss_mse.append(float(self.loss_mse(target, outputs)))
                loss_msle.append(float(self.loss_msle(target, outputs)))
        self.model_dev.train()
        return loss, loss_mae, loss_mse, loss_msle

    def fit(self, num_epochs, train_data, test_data, lr):
        optimizer = Adam(self.model_dev.parameters(), lr=lr)
        trainable_params, total_params = self.count_parameters(self.model_dev)
        print(f"Parameters: {trainable_params} trainable, {total_params} total")

        loss, loss_mae, loss_mse, loss_msle = self.for_test(test_data)
        self.writer.add_scalar('MAE loss for test (by epoch)', np.mean(loss_mae), 0)
        self.writer.add_scalar('MSE loss for test (by epoch)', np.mean(loss_mse), 0)
        self.writer.add_scalar('MSLE loss for test (by epoch)', np.mean(loss_msle), 0)

        for epoch in tqdm(range(1, num_epochs + 1)):
            train_loss_for_epoch = []
            loss_mae = []
            loss_mse = []
            loss_msle = []

            for epoch_batch_idx, data in enumerate(tqdm(train_data), 0):
                batch = data['image'].to(device)
                target = data['carbs'].to(device)

                optimizer.zero_grad()
                outputs = self.model_dev(batch)

                loss = self.calculate_loss(target, outputs)
                loss.backward()
                optimizer.step()

                train_loss_for_epoch.append(float(loss))
                loss_mae.append(float(self.loss_mae(target, outputs)))
                loss_mse.append(float(self.loss_mse(target, outputs)))
                loss_msle.append(float(self.loss_msle(target, outputs)))

            self.writer.add_scalar('MAE loss for train (by epoch)', np.mean(loss_mae), epoch)
            self.writer.add_scalar('MSE loss for train (by epoch)', np.mean(loss_mse), epoch)
            self.writer.add_scalar('MSLE loss for train (by epoch)', np.mean(loss_msle), epoch)

            loss, loss_mae_test, loss_mse_test, loss_msle_test = self.for_test(test_data)
            self.writer.add_scalar('MAE loss for test (by epoch)', np.mean(loss_mae_test), epoch)
            self.writer.add_scalar('MSE loss for test (by epoch)', np.mean(loss_mse_test), epoch)
            self.writer.add_scalar('MSLE loss for test (by epoch)', np.mean(loss_msle_test), epoch)

            self.model.save(self.model_dev, f'epoch_{epoch}', self.logdir)

        self.writer.close()
        print('done fitting epochs')


if __name__ == "__main__":
    args = get_args()
    model = Model(args,
                  batch_size=params['batch_size'])
    print('done with init model')

    train = FoodDataset(carbs_file='images/json_train_005.json',
                        image_dir='images/train_005',
                        type_data='train')
    train = DataLoader(train,
                       batch_size=params['batch_size'],
                       shuffle=True,
                       pin_memory=True)
    print('done with loading train data')

    test = FoodDataset(carbs_file='images/json_test_005.json',
                       image_dir='images/test_005',
                       type_data='test')
    test = DataLoader(test,
                      batch_size=params['batch_size'],
                      shuffle=True,
                      pin_memory=True)
    print('done with loading test data')
    model.fit(num_epochs=params['num_epochs'],
              train_data=train,
              test_data=test,
              lr=params['lr'])
