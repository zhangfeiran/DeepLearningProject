import os, sys

if sys.platform == 'linux':
    import pip
    pip.main(['install', 'pandas'])
    pip.main(['install', 'iterative-stratification'])
    pip.main(['install', 'imgaug'])
    os.chdir('./cos_person/')
    sys.path.append('./')

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import HPADataset

# hyper parameters
batch_size = 32

for fold_idx in range(4):
    if sys.platform == 'linux':
        path = './proteinatlas/'
    else:
        path = '../input/'

    train_dataset = HPADataset(path, fold_idx, 'train')
    val_dataset = HPADataset(path, fold_idx, 'val')
    test_dataset = HPADataset(path, None, 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    for i, (image,label) in enumerate(train_loader):
        print(image.shape,label.shape)
        break
    for i, (image,label) in enumerate(val_loader):
        print(image.shape,label.shape)
        break
    for i, data in enumerate(test_loader):
        print(data.shape)
        break

    break