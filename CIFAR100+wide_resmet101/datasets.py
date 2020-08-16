import os

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from .train import ROOT

transform = {
    "train": transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.2, 0.2),
            scale=(0.8, 1.2),
            shear=0.1
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}

train_dataset = datasets.CIFAR100(
    root=os.path.join(ROOT, "CIFAR100"),
    train=True,
    transform=transform['train'],
    download=True
)

test_dataset = datasets.CIFAR100(
    root=os.path.join(ROOT, "CIFAR100"),
    train=False,
    transform=transform['test'],
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)
