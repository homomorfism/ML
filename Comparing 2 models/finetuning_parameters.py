# Importing important libraries
# Importing important libraries
import gc

import matplotlib.pyplot as plt
import numpy as np

from .dataset import datasets, dataloaders
from .models import CNN, Resnet18
from .train import train

IM_SIZE = 28
classes = [i for i in range(10)]
ROOT = "/content/drive/My Drive/Colab Notebooks/MNIST fine-tuning"
device = "cuda"
np.set_printoptions(suppress=True)
gc.collect()


def imshow4(dataset):
    f = plt.figure()
    fig, ax = plt.subplots(1, 4, figsize=(10, 10))
    for i in range(4):
        ax[i].imshow(dataset[i][0][0], cmap='gray')
        ax[i].set_title(dataset[i][1])
        ax[i].axis("off")
        ax[i].plot()


train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']

print("Plotting images from train dataset")
imshow4(datasets['train'])

print("Plotting images from val dataset")
imshow4(datasets['val'])

print("Plotting images from test dataset")
imshow4(datasets['test'])

models = [CNN(), Resnet18()]

for net in models:
    train(net, train_loader, val_loader, test_loader)
