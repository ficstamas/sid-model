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

resnet = torchvision.models.resnet18(pretrained=True)

fpn = FPN(Bottleneck, [2, 2, 2, 2])
fan = FAN()
FAN_reg = FANRegression()
dataloader = DataLoader(augmented, 16, True)

for data in dataloader:
    fpn.train()
    fan.train()
    resnet.eval()

    x = data[0]
    features = fpn(x)
    attention = fan(features)
    cls = resnet(attention[0])

    cls2 = FAN_reg(cls)
    print("")
