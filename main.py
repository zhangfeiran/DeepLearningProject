import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import HPADataset

# hyper parameters
batch_size = 32

for fold_idx in range(4):
    dataset = HPADataset('../input/', fold_idx, 'train')
    train_loader = DataLoader(dataset, batch_size=batch_size)

    for i,data in enumerate(train_loader):
        print(data[0].shape)
        break

    break