import torch.nn as nn
from torchvision import models

net = models.wide_resnet101_2(pretrained=True)
net.classifier = nn.Linear(1024, 100)

for layer in [net.layer1, net.layer2, net.layer3]:
    for param in layer.parameters():
        param.requires_grad_(False)

# net = models.wide_resnet101_2(pretrained=True)
# net.classifier = nn.Sequential(
#     nn.Dropout(0.5),
#     nn.Linear(1024, 100)
# )