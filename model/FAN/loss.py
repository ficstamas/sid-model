from torch.nn.functional import interpolate
import torch.nn as nn
import torch


def focal_loss(actual, predict):
    pass


def attention_loss(actual, predict, bce: nn.BCEWithLogitsLoss):
    actual = torch.reshape(actual, shape=(16, 1, 512, 512))
    down_sample_sizes = [128, 64, 32, 16]
    target = []
    for i, size in enumerate(down_sample_sizes):
        target.append(interpolate(actual, size=[size, size], mode="nearest"))

    loss = 0

    for i in range(4):
        loss += bce(predict[i], target[i])
    return loss
