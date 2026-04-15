"""
U-Net: Convolutional Networks for Biomedical Image Segmentation
Ronneberger et al., 2015 (https://arxiv.org/abs/1505.04597)

논문 원본 스펙:
- 입력: 572×572 (논문), 임의 크기 가능 (padding 덕분)
- 출력: 388×388 (논문), valid conv 기준 / 본 구현은 same conv로 동일 크기 유지
- Contracting path: Conv 3×3 (unpadded) × 2 → MaxPool 2×2 (채널 2배, 해상도 절반)
- Expansive path:   UpConv 2×2 → skip concat → Conv 3×3 × 2
- 최종 출력: 1×1 Conv → 클래스 수 채널
- Weight init: He (kaiming) initialization (논문 언급)
- No Dropout (논문 원본), BatchNorm 없음 (논문 원본)
  → 본 구현은 학습 안정성을 위해 BatchNorm 추가 (optional 파라미터)
"""

import torch
import torch.nn as nn


def double_conv(in_ch: int, out_ch: int, use_bn: bool = True) -> nn.Sequential:
    """
    논문 Figure 1의 'conv 3x3, ReLU' × 2 블록
    padding=1 → same convolution (원본은 valid이나 실용적으로 same 사용)
    """
    layers = []
    layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))

    layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class EncoderBlock(nn.Module):
    """
    Contracting Path의 한 단계
    double_conv → MaxPool (2×2)
    forward()는 pool 전후 feature 모두 반환
    → skip connection을 위해 pool 전 feature 보존
    """
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch, use_bn)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feat = self.conv(x)   # skip connection에 사용
        pooled = self.pool(feat)
        return feat, pooled


class DecoderBlock(nn.Module):
    """
    Expansive Path의 한 단계
    논문: 'up-conv 2×2' → skip concat → double_conv
    ConvTranspose2d가 논문의 up-convolution
    """
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        # up-conv: 해상도 2배, 채널 절반
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # concat 후 채널이 out_ch*2 → double_conv로 out_ch로 압축
        self.conv = double_conv(out_ch * 2, out_ch, use_bn)

    def forward(self, x, skip):
        x = self.upconv(x)
        # skip과 크기가 다를 경우 center crop (논문 방식)
        if x.shape != skip.shape:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = nn.functional.pad(x, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2
            ])
        x = torch.cat([skip, x], dim=1)  # 논문: copy and crop
        return self.conv(x)


class UNet(nn.Module):
    """
    논문 Figure 1 기준 채널 구성:
    Encoder: 1 → 64 → 128 → 256 → 512
    Bottleneck:     512 → 1024
    Decoder: 1024 → 512 → 256 → 128 → 64
    Output:  64 → n_classes (1×1 Conv)

    Args:
        in_channels  : 입력 채널 수 (논문: 1, grayscale)
        out_channels : 출력 클래스 수 (논문: 2, binary segmentation)
        features     : 첫 번째 encoder의 채널 수 (논문: 64)
        use_bn       : BatchNorm 사용 여부 (논문 원본 False, 실용적으로 True 권장)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: int = 64,
        use_bn: bool = True,
    ):
        super().__init__()

        # ── Contracting Path (Encoder) ──────────────────────────────────────
        self.enc1 = EncoderBlock(in_channels,      features,     use_bn)  # 1→64
        self.enc2 = EncoderBlock(features,         features * 2, use_bn)  # 64→128
        self.enc3 = EncoderBlock(features * 2,     features * 4, use_bn)  # 128→256
        self.enc4 = EncoderBlock(features * 4,     features * 8, use_bn)  # 256→512

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = double_conv(features * 8, features * 16, use_bn)  # 512→1024

        # ── Expansive Path (Decoder) ─────────────────────────────────────────
        self.dec4 = DecoderBlock(features * 16,  features * 8, use_bn)   # 1024→512
        self.dec3 = DecoderBlock(features * 8,   features * 4, use_bn)   # 512→256
        self.dec2 = DecoderBlock(features * 4,   features * 2, use_bn)   # 256→128
        self.dec1 = DecoderBlock(features * 2,   features,     use_bn)   # 128→64

        # ── Final 1×1 Conv ────────────────────────────────────────────────────
        # 논문: "1×1 convolution is used to map each 64-component feature vector
        #        to the desired number of classes"
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

        # ── Weight Initialization ─────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        논문 인용: "we initialize the weights ... with the standard deviation of
        sqrt(2/N)" → He (kaiming) initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Contracting
        skip1, x = self.enc1(x)   # skip1: 64ch
        skip2, x = self.enc2(x)   # skip2: 128ch
        skip3, x = self.enc3(x)   # skip3: 256ch
        skip4, x = self.enc4(x)   # skip4: 512ch

        # Bottleneck
        x = self.bottleneck(x)    # 1024ch

        # Expansive + skip connections
        x = self.dec4(x, skip4)   # 512ch
        x = self.dec3(x, skip3)   # 256ch
        x = self.dec2(x, skip2)   # 128ch
        x = self.dec1(x, skip1)   # 64ch

        # Output (logit, sigmoid/softmax는 loss 함수에서 처리)
        return self.final_conv(x)


if __name__ == '__main__':
    # 빠른 검증: 입출력 shape 확인
    model = UNet(in_channels=1, out_channels=1, features=64, use_bn=True)
    x = torch.randn(2, 1, 572, 572)   # 논문 입력 크기
    out = model(x)
    print(f"입력:  {x.shape}")        # (2, 1, 572, 572)
    print(f"출력:  {out.shape}")      # (2, 1, 572, 572)

    total = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {total:,}")   # ~31M
