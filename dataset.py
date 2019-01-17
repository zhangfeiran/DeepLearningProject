import os
import torch
import numpy as np
import pandas as pd
import scipy.misc as misc
import cv2
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class HPADataset(object):
    def __init__(self, path: str, idx: int, split: str):
        self.path = path
        df = pd.read_csv('input/train.csv')
        df['target_vec'] = df['Target'].map(lambda x: list(map(int, x.strip().split())))
        df['onehot'] = df['target_vec'].map(lambda x: np.sum(np.eye(28, dtype=int)[x], axis=0))
        X = df['Id'].tolist()
        y = df['onehot'].tolist()
        mskf = MultilabelStratifiedKFold(n_splits=4, random_state=0)
        for i, (train, val) in enumerate(mskf.split(X, y)):
            if i == idx:
                self.df = df.iloc[train] if split == 'train' else df.iloc[val]
        self.len = len(self.df)

    def transform(self):
        pass

    def __getitem__(self, index):
        pass
        data = self.df.iloc[index]
        name = [data['Id'] + "_" + color + ".png" for color in ["red", "green", "blue"]]
        images = [cv2.imread(self.path + i, cv2.IMREAD_GRAYSCALE) for i in name]
        image = img = np.stack(images, axis=2)
        label = np.array(data['onehot'])
        return (image, label)

    def __len__(self):
        return self.len


if __name__ == '__main__':
    HPADataset('../input/train2/', 0, 'train')
