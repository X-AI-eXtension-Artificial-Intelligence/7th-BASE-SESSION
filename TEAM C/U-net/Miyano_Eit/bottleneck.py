import torch.nn as nn
from encoder import DoubleConv


class Bottleneck(nn.Module):
    """
    논문의 가장 아래 레이어 (Bridge)
    Encoder와 Decoder 사이를 연결
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)