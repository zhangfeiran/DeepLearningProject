import os
import torch
import numpy as np
import pandas as pd
import scipy.misc as misc
import cv2
import torchvision as tv
from torch.utils.data import Dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from imgaug import augmenters as iaa
from torchvision import transforms as T
from get_sample_weights import get_sample_weights


class HPADataset(Dataset):
    def __init__(self, path: str, idx: int, split: str):
        self.path = path
        self.split = split
        self.get_augumentor(split)
        if split == 'test':
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
            self.weights = get_sample_weights(self.df)
        if __name__ == '__main__':
            print(self.df)
            self.__getitem__(0)

    def transform(self, img):
        return self.augumentor(img)

    def get_augumentor(self, split):
        if split != 'train':
            self.augumentor = T.Compose(
                [T.ToPILImage(), T.ToTensor(),
                 T.Normalize([0.0789, 0.0529, 0.0546, 0.0814], [0.147, 0.113, 0.157, 0.148])])
            return

        aug = iaa.SomeOf(
            n=(1, 6), children=[
                iaa.Noop(),
                iaa.Sequential([iaa.Add((-5, 5), per_channel=True),
                                iaa.Multiply((0.8, 1.2), per_channel=True)]),
                iaa.Crop(percent=(0, 0.15)),
                iaa.Affine(shear=(-16, 16)),
                iaa.OneOf([
                    iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270),
                    iaa.Fliplr(1),
                    iaa.Flipud(1),
                ])
            ])

        self.augumentor = T.Compose(
            [aug.augment_image,
             T.ToPILImage(),
             T.ToTensor(),
             T.Normalize([0.0789, 0.0529, 0.0546, 0.0814], [0.147, 0.113, 0.157, 0.148])])

    def __getitem__(self, index):
        index = int(index)
        print(index)
        data = self.df.iloc[index]
        name = [data['Id'] + "_" + color + ".png" for color in ["red", "green", "blue", "yellow"]]
        images = [cv2.imread(f"{self.path}{'test' if self.split=='test' else 'train'}/{i}", cv2.IMREAD_GRAYSCALE) for i in name]

        image = np.stack(images, axis=-1)
        image = self.transform(image)

        if __name__ == '__main__':
            print(image.shape)

        if self.split == 'test':
            image = image.numpy()
            images = [image]
            images.append(np.fliplr(image))
            images.append(np.flipud(image))
            images.append(np.fliplr(images[-1]))
            images.append(np.transpose(image, (0, 2, 1)))
            images.append(np.flipud(images[-1]))
            images.append(np.fliplr(images[-2]))
            images.append(np.flipud(images[-1]))
            images = np.stack(images, axis=0)
            return {'image': torch.Tensor(images)}

        label = np.array(data['onehot'])
        return {'image': image, 'label': label}

    def __len__(self):
        return self.len


if __name__ == '__main__':
    HPADataset('../input/', 0, 'train')
