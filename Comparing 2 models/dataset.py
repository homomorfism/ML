import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from .finetuning_parameters import ROOT


IM_SIZE = 28


class MNIST(Dataset):
    """
    Custom MNIST Dataset class, which allows to read data from csv-file
    """

    def __init__(self, csv_path, transform, shuffle_):
        df = pd.read_csv(csv_path)

        if shuffle_:
            df = df.sample(frac=1, random_state=3)

        self.data = df.values

        self.transform = transform

        self.labels = [i for i in range(10)]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = self.data[index, 1:].reshape(IM_SIZE, IM_SIZE)
        label = int(self.data[index, 0])

        image = torch.as_tensor(image, dtype=torch.float)
        image = torch.stack([image, image, image])

        image = self.transform(image)

        return image, label

BATCH_SIZE = 1024
transform = {
    "train": transforms.Compose([
        transforms.ToPILImage(mode="RGB"),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.25, 0.25),
            scale=(0.7, 1.3),
            shear=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    "test": transforms.Compose([
        transforms.ToPILImage(mode="RGB"),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}

datasets = {
    "train": MNIST(
        csv_path=os.path.join(ROOT, "Kaddana-MNIST/train.csv"),
        transform=transform["train"],
        shuffle_=True
    ),
    "val": MNIST(
        csv_path=os.path.join(ROOT, "Kaddana-MNIST/Dig-MNIST.csv"),
        transform=transform["test"],
        shuffle_=True
    ),
    "test": MNIST(
        csv_path=os.path.join(ROOT, "MNIST/train.csv"),
        transform=transform["test"],
        shuffle_=True
    )
}
dataloaders = {
    "train": DataLoader(
        dataset=datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8
    ),
    "val": DataLoader(
        dataset=datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8
    ),
    "test": DataLoader(
        dataset=datasets['test'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8
    )
}