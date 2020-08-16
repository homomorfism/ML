import torch.nn as nn

class CNN_32_32(nn.Module):
    def __init__(self):
        super(CNN_32_32, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=5,
            padding=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            padding=2
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2
        )

        self.relu1 = nn.ReLU()

        self.drop1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2
        )
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            padding=2
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=2
        )

        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(4096, 256)
        self.fc_relu1 = nn.ReLU()
        self.fc_drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool1(out)
        out = self.drop1(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool2(out)
        out = self.drop2(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc_relu1(out)
        out = self.fc_drop1(out)

        out = self.fc2(out)

        return out

