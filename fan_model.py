from model.data.dataset import MillionFacesDataset, FANDataset
from model.data.augment import ComposeFANPortrait
from torch.utils.data import DataLoader
from model.FAN.fan import *
from model.FAN.loss import attention_loss
from torch.optim import SGD
import tqdm
import torch
import numpy as np
from sklearn.metrics import f1_score
import json

import os

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  # "cuda:1" if torch.cuda.is_available() else

# 512 x 512 input size

prefix = "/data/takacse/fan/"
train_data_path = f"{prefix}/train/"
test_data_path = f"{prefix}/test/"
epochs = 100

# create dirs

os.makedirs(os.path.join(prefix, "model"), exist_ok=True)

with torch.cuda.device(2) as d:
    # Data Train
    train = MillionFacesDataset(train_data_path)
    augmented = FANDataset(train, transform=ComposeFANPortrait(25, 0.8, 0.5))
    dataloader = DataLoader(augmented, 16, True)
    # Data Test
    test = MillionFacesDataset(test_data_path)
    augmented_test = FANDataset(test, transform=ComposeFANPortrait(25, 0.8, 0.5))
    dataloader_test = DataLoader(augmented_test, 16, True)

    fpn = FPN(Bottleneck, [2, 2, 2, 2]).cuda(d)
    FAN_reg = FANRegression().cuda(d)
    fan = FAN(fpn, FAN_reg).cuda(d)

    criterion = torch.nn.BCEWithLogitsLoss().cuda(d)
    optimizer = SGD(fan.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

    reg_loss = nn.BCELoss().cuda(d)

    epoch_data = {"train_loss": [], "test_score": []}

    for epoch in tqdm.trange(epochs, desc="Epoch"):
        fpn.train()
        fan.train()
        train_loss = []
        for data in tqdm.tqdm(dataloader, desc="Train"):
            image = data[0].cuda(d)
            mask = data[1].cuda(d)
            label = data[2].cuda(d)

            out, attention, prediction = fan(image)
            loss = attention_loss(mask, attention, criterion) + reg_loss(torch.argmax(prediction, dim=1).float(), label.resize(16).float())
            train_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_data["train_loss"].append(train_loss)

        fan.eval()

        with torch.no_grad():
            test_predictions = np.array([])
            test_labels = np.array([])
            for data in tqdm.tqdm(dataloader_test, desc="Test"):
                image = data[0].cuda(d)
                mask = data[1].cuda(d)
                label = data[2].resize(16).long().cuda(d)

                out, attention, prediction = fan(image)
                pred = torch.argmax(prediction, dim=1).long()
                test_predictions = np.concatenate([test_predictions, pred.cpu().numpy()])
                test_labels = np.concatenate([test_labels, label.cpu().numpy()])
            epoch_data["test_score"].append(f1_score(test_labels, test_predictions))

        torch.save(fan, f"{prefix}model/fan_{epoch}.pt")
        with open(f"{prefix}model/epochs.json", mode="w", encoding="utf8") as f:
            json.dump(epoch_data, f)
