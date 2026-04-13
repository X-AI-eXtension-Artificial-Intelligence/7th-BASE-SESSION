"""
U-Net: Convolutional Networks for Biomedical Image Segmentation
Ronneberger et al., 2015 - arXiv:1505.04597

논문 기반 U-Net 구현 (PyTorch)
- Contracting path: 3x3 conv x2 + ReLU + 2x2 max pool (채널 2배)
- Bottleneck: 3x3 conv x2 + ReLU (1024채널)
- Expansive path: 2x2 up-conv + skip connection(concat) + 3x3 conv x2 + ReLU
- Final: 1x1 conv → num_classes
- 가중치 초기화: He initialization (논문 Section 3)
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """두 번의 3x3 Conv + BN + ReLU 블록 (논문: 3x3 conv, unpadded → 여기서는 padding=1로 출력 크기 유지)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net 아키텍처 (논문 Figure 1 기반)

    구조:
      Encoder: [64 → 128 → 256 → 512] + 2x2 MaxPool
      Bottleneck: 1024
      Decoder: [512 → 256 → 128 → 64] + 2x2 UpConv + Skip connection
      Output: 1x1 Conv → num_classes
    """
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()

        # Contracting path (Encoder)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Expansive path (Decoder)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Final 1x1 conv
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        # He initialization (논문 Section 3: std = sqrt(2/N))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)               # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))   # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))   # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))   # [B, 512, H/8, W/8]

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # [B, 1024, H/16, W/16]

        # Decoder (skip connection: concatenation)
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))    # [B, 512, H/8, W/8]
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))   # [B, 256, H/4, W/4]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))   # [B, 128, H/2, W/2]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))   # [B, 64, H, W]

        return self.final(d1)  # [B, num_classes, H, W]


if __name__ == '__main__':
    model = UNet(in_channels=3, num_classes=2)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total_params:,}")
