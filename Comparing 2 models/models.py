import torch.nn as nn
from torchvision import models


def conv_block(in_channels, out_channels, stride, ks):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ks,
            padding=1,
            stride=stride
        ),
        nn.BatchNorm2d(
            num_features=out_channels,
            momentum=0.9,
            eps=1e-5
        ),
        nn.LeakyReLU(
            negative_slope=0.1
        )
    )


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            conv_block(3, 32, stride=1, ks=5),
            conv_block(32, 32, stride=1, ks=5),
            conv_block(32, 32, stride=2, ks=8),
            nn.Dropout(0.4),

            conv_block(32, 64, stride=1, ks=5),
            conv_block(64, 64, stride=1, ks=5),
            conv_block(64, 64, stride=2, ks=5),
            nn.Dropout(0.4),

            nn.Flatten(),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.1),

            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

    def get_name(self):
        return "CNN"


class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(pretrained=True)

        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, 10)

    def forward(self, x):
        return self.net(x)

    def get_name(self):
        return "resnet18"
