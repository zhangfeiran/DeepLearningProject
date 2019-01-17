import os
import torch
import numpy as np
import pandas as pd
import scipy.misc as misc
import cv2
import torchvision as tv
from torch.utils.data import Dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class HPADataset(Dataset):
    def __init__(self, path: str, idx: int, split: str):
        self.path = path
        self.eval = False
        if split == 'test':
            self.eval = True
            self.df = pd.read_csv(self.path + 'sample_submission.csv')
            self.len = len(self.df)
        else:
            df = pd.read_csv(self.path + 'train.csv')
            df['target_vec'] = df['Target'].map(lambda x: list(map(int, x.strip().split())))
            df['onehot'] = df['target_vec'].map(lambda x: np.sum(np.eye(28, dtype=int)[x], axis=0))
            X = df['Id'].tolist()
            y = df['onehot'].tolist()
            mskf = MultilabelStratifiedKFold(n_splits=4, random_state=0)
            for i, (train, val) in enumerate(mskf.split(X, y)):
                if i == idx:
                    self.df = df.iloc[train] if split == 'train' else df.iloc[val]
            self.len = len(self.df)
        if __name__ == '__main__':
            self.__getitem__(0)

    def transform(self, img):
        return tv.transforms.ToTensor()(img)

    def __getitem__(self, index):
        pass
        data = self.df.iloc[index]
        name = [data['Id'] + "_" + color + ".png" for color in ["red", "green", "blue", "yellow"]]
        images = [cv2.imread(self.path + 'train/' + i, cv2.IMREAD_GRAYSCALE) for i in name]

        image = np.stack(images, axis=-1)
        image = self.transform(image)

        if __name__ == '__main__':
            print(image.shape)
            print(type(image[0, 0, 0]))

        if self.eval:
            return image

        label = np.array(data['onehot'])
        return image, label

    def __len__(self):
        return self.len


if __name__ == '__main__':
    HPADataset('../input/', 0, 'train')
