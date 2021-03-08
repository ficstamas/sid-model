from model.data.dataset import MillionFacesDataset, FANDataset
import matplotlib.pyplot as plt
from model.data.augment import ComposeFANPortrait
import numpy as np
import tqdm

# np.random.seed(1)

# 512 x 512 input size

ds = MillionFacesDataset("data/million/")
augmented = FANDataset(ds, transform=ComposeFANPortrait(25, 0.8, 0.5))

i= 0
try:
    for img, mask in tqdm.tqdm(augmented):
        continue
except Exception as ex:
    print(ex)

