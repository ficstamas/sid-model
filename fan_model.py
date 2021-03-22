from model.data.dataset import MillionFacesDataset, FANDataset
from model.data.augment import ComposeFANPortrait
import torchvision
from torch.utils.data import DataLoader
from model.FAN.fan import *
from model.FAN.loss import attention_loss
from torch.optim import SGD
import tqdm
import torch
import numpy as np

# 512 x 512 input size

# Data Train
train = MillionFacesDataset("data/million/")
augmented = FANDataset(train, transform=ComposeFANPortrait(25, 0.8, 0.5))
dataloader = DataLoader(augmented, 16, True)
# Data Test
test = MillionFacesDataset("data/test/")
augmented_test = FANDataset(test, transform=ComposeFANPortrait(25, 0.8, 0.5))
dataloader_test = DataLoader(augmented_test, 16, True)


# resnet = torchvision.models.resnet18(pretrained=True)

fpn = FPN(Bottleneck, [2, 2, 2, 2])
fan = FAN()
# FAN_reg = FANRegression()
# resnet.eval()

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = SGD(fpn.parameters(), lr=0.01, momentum=0.9)
optimizer2 = SGD(fan.parameters(), lr=0.01, momentum=0.9)
# criterion = MSELoss()
# cls = resnet(out[0])
# cls2 = FAN_reg(cls)

for epoch in tqdm.trange(100, desc="Epoch"):
    fpn.train()
    fan.train()
    for data in tqdm.tqdm(dataloader, desc="Train"):
        image = data[0]
        mask = data[1]
        label = data[2]

        features = fpn(image)
        out, attention = fan(features)
        loss = attention_loss(mask, attention, criterion)

        optimizer.zero_grad()
        optimizer2.zero_grad()

        loss.backward()

        optimizer.step()
        optimizer2.step()

    fpn.eval()
    fan.eval()

    with torch.no_grad():
        test_loss = []
        for data in tqdm.tqdm(dataloader_test, desc="Test"):
            image = data[0]
            mask = data[1]
            label = data[2]

            features = fpn(image)
            out, attention = fan(features)
            loss = attention_loss(mask, attention, criterion)
            test_loss.append(loss.detach().numpy())
        print("Test loss", np.average(test_loss))
    torch.save(fpn, "data/model/fpn.pt")
    torch.save(fan, "data/model/fan.pt")
