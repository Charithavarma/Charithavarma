#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from model import UNET
from utils import get_data_loaders, calculate_accuracy
from config import hp


def train(pre_trained=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    train_loader = get_data_loaders(hp=hp, data_type='train', batch_size=hp['batch_size'], shuffle=False,
                                    drop_last=False,
                                    num_workers=hp['num_workers'], pin_memory=True)
    val_loader = get_data_loaders(hp=hp, data_type='val', batch_size=hp['batch_size'], shuffle=False, drop_last=False,
                                  num_workers=hp['num_workers'], pin_memory=True)

    # model
    model = UNET().to(device).double()

    # loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])

    epochs = hp['num_epochs']

    # training loop
    for epoch in range(epochs):
        epoch_train_loss = []
        epoch_val_loss = []

        dice_score_train = 0
        dice_score_val = 0

        model.train()

        for i, (image, mask) in enumerate(train_loader):
            image = image.to(device, dtype=torch.double)
            mask = mask.to(device, dtype=torch.double)

            optimizer.zero_grad()
            preds = model(image)

            loss = loss_fn(preds, mask)
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())
            dice_score_train = calculate_accuracy(preds=preds, targets=mask)

        model.eval()

        with torch.no_grad():
            for i, (image, mask) in enumerate(val_loader):
                image = image.to(device, dtype=torch.double)
                mask = mask.to(device, dtype=torch.double)

                preds = model(image)

                loss = loss_fn(preds, mask)
                epoch_val_loss.append(loss.item())
                dice_score_val = calculate_accuracy(preds=preds, targets=mask)

        # Calculate mean losses
        epoch_train_loss = np.array(epoch_train_loss).mean()
        epoch_val_loss = np.array(epoch_val_loss).mean()

        print("epoch = {}, Training loss = {}, Dice_score_train = {}, Validation loss = {} Dice Score_Val = {}"
              .format(epoch, epoch_train_loss, dice_score_train, epoch_val_loss, dice_score_val))


if __name__ == '__main__':
    train()
