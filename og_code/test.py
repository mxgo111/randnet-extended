import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
from sparselandtools.dictionaries import DCTDictionary
import os
from datetime import datetime
from sacred import Experiment
import torchvision
import generator
import sys

batch_size = 15

train_loader = generator.get_MNIST_loader(batch_size)
# if we want training, shuffle=True
for image, label in train_loader:
    y = image
    print("old y shape", y.shape)
    flat = y.view(-1, 1, 28*28)
    print("new y shape", flat.shape)
    unflatten = flat.view(-1, 1, 28, 28)
    print("unflatten shape", unflatten.shape)
    print(y - flat)
    # print(label.shape)
    sys.exit()


# TODO: this experiment but for rectangular matrix

# for view, -1 usually used for batch size
