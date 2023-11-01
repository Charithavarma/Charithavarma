#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import Dataset
import os
import nibabel as nib
from scipy.ndimage.interpolation import rotate
from config import hp

__docformat__ = 'reStructuredText'
__all__ = ['AbideDataset']


class AbideDataset(Dataset):
    def __init__(self, hp, data_type, transform=None):
        super(AbideDataset, self).__init__()
        self.hp = hp
        self.data_type = data_type
        self.transform = transform

        # self.angles = [5, 10, 45, -45, 90, 180, 270, ]
        # self.flips = [1, -1]
        self.data = []
        # self.masks = []
        self.load_data(hp['data_path'])

        print(len(self.data))
        # print(len(self.masks))

        # data split (train, test)
        # self.train_data = self.data[:int(0.75 * len(self.data))]
        # self.test_data = self.data[int(0.75 * len(self.data)):]
        # self.train_masks = self.masks[:int(0.75 * len(self.data))]
        # self.test_masks = self.masks[int(0.75 * len(self.data)):]

    def load_data(self, root_dir):
        #for root, subdirs, files in os.walk(root_dir):
         #   if len(files) > 0:
        image = nib.load(os.path.join(root, 'abide_msp.nii')).get_fdata()
        mask = nib.load(os.path.join(root, 'abide_cc.nii')).get_fdata()
        self.data.append(image)
                # self.masks.append(mask)

    def __len__(self):
        if self.data_type == 'train':
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, item):
        if self.data_type == 'train':
            self.image = np.array(self.train_data[item])
            self.mask = np.array(self.train_masks[item])
            self.mask[self.mask == 255] = 1
        else:
            self.image = np.array(self.test_data[item])
            self.mask = np.array(self.test_masks[item])
            self.mask[self.mask == 255] = 1

        image = np.transpose(self.image, (2, 0, 1))
        mask = np.transpose(self.mask, (2, 0, 1))

        return image, mask


if __name__ == '__main__':
    obj = AbideDataset(hp=hp, data_type='test')
