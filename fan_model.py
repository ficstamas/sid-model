from model.data.dataset import MillionFacesDataset, FANDataset
from model.data.augment import ComposeFANPortrait
import torch
import torchvision
from torch.utils.data import DataLoader
from model.FAN.fan import *
import numpy as np


# 512 x 512 input size

ds = MillionFacesDataset("data/million/")
augmented = FANDataset(ds, transform=ComposeFANPortrait(25, 0.8, 0.5))

# resnet = torchvision.models.resnet18(pretrained=True)

fpn = FPN(Bottleneck, [2, 2, 2, 2])

dataloader = DataLoader(augmented, 16, True)

for data in dataloader:
    fpn.train()
    x = data[0]
    x = fpn(x)
    y = x
