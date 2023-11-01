#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch

from dataset import AbideDataset
from torch.utils.data import DataLoader

__docformat__ = 'reStructuredText'
__all__ = ['get_data_loaders']


def get_data_loaders(hp, data_type: str,
                     batch_size: int,
                     shuffle: bool,
                     drop_last: bool,
                     pin_memory: bool,
                     num_workers: int) -> DataLoader:
    dataset = AbideDataset(hp=hp, data_type=data_type)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                      drop_last=drop_last, num_workers=num_workers)


def calculate_accuracy(preds, targets):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    targets = targets.unsqueeze(1)

    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    # pred_vals, pred_inds = torch.max(preds.data, 1)
    num_correct += (preds == targets)
    num_pixels += torch.numel(preds)

    dice_score += (2 * (preds * targets).sum()) / ((preds + targets).sum() + 1e-8)

    # preds = preds.argmax(1).cpu().numpy()
    # targets = targets.argmax(1).cpu().numpy()
    # intersection = np.logical_and(targets, preds)
    # union = np.logical_or(targets, preds)
    # iou_score = np.sum(intersection) / np.sum(union)
    # print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")

    # accuracy = (num_correct / num_pixels) * 100

    return dice_score

# EOF
