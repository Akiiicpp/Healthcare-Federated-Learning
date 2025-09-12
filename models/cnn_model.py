from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + identity)
        return x


class SmallResNet18(nn.Module):

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = nn.Sequential(
            SmallResNetBlock(64, 64),
            SmallResNetBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            SmallResNetBlock(64, 128, stride=2),
            SmallResNetBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            SmallResNetBlock(128, 256, stride=2),
            SmallResNetBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            SmallResNetBlock(256, 512, stride=2),
            SmallResNetBlock(512, 512),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x


def create_model(in_channels: int = 1, num_classes: int = 2) -> nn.Module:
    return SmallResNet18(in_channels=in_channels, num_classes=num_classes)
