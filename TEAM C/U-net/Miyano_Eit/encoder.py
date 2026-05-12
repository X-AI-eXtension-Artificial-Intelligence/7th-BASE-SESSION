import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    논문의 기본 빌딩 블록
    Conv 3×3 → BN → ReLU → Conv 3×3 → BN → ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    논문의 Contracting Path
    DoubleConv → MaxPool2d
    skip connection을 위해 MaxPool 이전 feature map도 반환
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.conv(x)     # skip connection으로 사용할 feature map
        pooled = self.pool(skip)
        return pooled, skip