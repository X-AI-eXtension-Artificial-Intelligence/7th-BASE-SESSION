import torch
import torch.nn as nn
from encoder import EncoderBlock
from bottleneck import Bottleneck
from decoder import DecoderBlock


class UNet(nn.Module):
    """
    논문: U-Net: Convolutional Networks for Biomedical Image Segmentation
    Ronneberger et al. (2015)

    구조:
    Encoder (Contracting Path): 특징 추출 + 공간 정보 압축
    Bottleneck:                 가장 압축된 표현
    Decoder (Expansive Path):   공간 정보 복원 + skip connection

    입력:  (batch, in_channels, H, W)
    출력:  (batch, num_classes, H, W)  → 픽셀별 클래스 확률
    """
    def __init__(self, in_channels=3, num_classes=2,
                 features=[64, 128, 256, 512]):
        super().__init__()

        # Encoder: 논문의 Contracting Path
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(ch, f))
            ch = f

        # Bottleneck: 가장 아래 레이어
        self.bottleneck = Bottleneck(features[-1], features[-1] * 2)

        # Decoder: 논문의 Expansive Path (역순)
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(DecoderBlock(f * 2, f))

        # 최종 출력: 1×1 Conv로 클래스 수로 변환
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoding: 각 단계의 skip connection 저장
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoding: skip connection을 역순으로 사용
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.final_conv(x)