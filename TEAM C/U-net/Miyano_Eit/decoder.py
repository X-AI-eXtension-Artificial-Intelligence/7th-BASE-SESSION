import torch
import torch.nn as nn
from encoder import DoubleConv


class DecoderBlock(nn.Module):
    """
    논문의 Expansive Path
    Upsample → skip connection과 concatenate → DoubleConv

    논문 핵심: skip connection으로 encoder의 feature map을 가져와
    공간 정보(spatial information)를 복원함
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ConvTranspose2d: 학습 가능한 업샘플링 (논문 방식)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                           kernel_size=2, stride=2)
        # skip connection과 concat 후 채널이 2배 → DoubleConv로 줄임
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)

        # 크기가 맞지 않을 때 center crop (논문 방식)
        if x.shape != skip.shape:
            x = self._crop(x, skip)

        x = torch.cat([skip, x], dim=1)    # channel 차원으로 concat
        return self.conv(x)

    def _crop(self, x, skip):
        # skip이 x보다 클 경우 중앙 crop
        _, _, H, W = x.shape
        _, _, sH, sW = skip.shape
        dH = (sH - H) // 2
        dW = (sW - W) // 2
        skip = skip[:, :, dH:dH+H, dW:dW+W]
        return x