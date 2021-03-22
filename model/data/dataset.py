from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from skimage import io
import torch
import os
import numpy as np
from .augment import ComposeFANInput
import matplotlib.image as plt_image


class MillionFacesDataset(Dataset):
    def __getitem__(self, index) -> T_co:
        if torch.is_tensor(index):
            index = index.tolist()

        image = plt_image.imread(self.images[index])

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)

    def __init__(self, path: str, transform=None):
        super(MillionFacesDataset, self).__init__()

        self.transform = transform

        self.images = []
        for dirpath, _, filenames in os.walk(path):
            for file in filenames:
                if file.endswith(".png") or file.endswith(".jpg"):
                    self.images.append(os.path.join(dirpath, file))


class FANDataset(Dataset):
    def __getitem__(self, index) -> T_co:
        if torch.is_tensor(index):
            index = index.tolist()

        sample = self.dataset[index]
        r = np.random.random(1)
        label = True

        if r < self.p:
            ridx = np.random.randint(0, high=len(self))
            portrait = self.dataset[ridx]
            label = index == ridx
        else:
            portrait = self.dataset[index]

        compose = ComposeFANInput(self.size, self.portrait_scale, self.transform)
        image, mask = compose(sample, portrait)
        return image, mask, torch.BoolTensor([label])

    def __len__(self):
        return self.dataset.__len__()

    def __init__(self, dataset: MillionFacesDataset, size=(512, 512), portrait_scale=8, transform=None, p=0.5):
        super(FANDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.size = size
        self.portrait_scale = portrait_scale
        self.p = p

