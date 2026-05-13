import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))


class DarkNet(nn.Module):
    """
    논문 구조 대신 toy용 경량 버전
    24 Conv → 5 Conv로 축소
    """
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

        self.features = nn.Sequential(
            ConvBlock(3, 32, 3, stride=2, padding=1),    # 112 → 56
            nn.MaxPool2d(2, 2),                           # 56 → 28

            ConvBlock(32, 64, 3, padding=1),              # 28 → 28
            nn.MaxPool2d(2, 2),                           # 28 → 14

            ConvBlock(64, 128, 3, padding=1),             # 14 → 14
            nn.MaxPool2d(2, 2),                           # 14 → 7

            ConvBlock(128, 256, 3, padding=1),            # 7 → 7
            ConvBlock(256, 256, 3, padding=1),            # 7 → 7
        )

        # 어떤 입력 크기든 S×S로 고정
        self.adaptive_pool = nn.AdaptiveAvgPool2d((S, S))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * S * S, 1024),               # 4096 → 1024
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(1024, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        x = torch.sigmoid(x)
        return x