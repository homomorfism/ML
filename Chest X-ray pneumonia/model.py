from torchvision import models
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(pretrained=True)

        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.net(x)
