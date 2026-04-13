import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation

    핵심 구조:
    - Contracting Path: 의미 정보 추출 (해상도 ↓, 채널 ↑)
    - Expansive Path: 위치 정보 복원 (해상도 ↑)
    - Skip Connection: Contracting의 feature를 Expansive로 전달하여
                       semantic + localization 정보 결합
    """

    def __init__(self):
        super(UNet, self).__init__()

        # 기본 블록: Conv → BN → ReLU
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)]
            layers += [nn.BatchNorm2d(out_channels)]
            layers += [nn.ReLU()]
            return nn.Sequential(*layers)

        # =========================
        # Contracting Path
        # 이미지에서 점점 더 추상적인 feature 추출
        # 해상도 ↓ / 채널 ↑

        self.enc1_1 = CBR2d(1, 64)
        self.enc1_2 = CBR2d(64, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = CBR2d(64, 128)
        self.enc2_2 = CBR2d(128, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = CBR2d(128, 256)
        self.enc3_2 = CBR2d(256, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4_1 = CBR2d(256, 512)
        self.enc4_2 = CBR2d(512, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck (가장 압축된 표현)
        self.enc5_1 = CBR2d(512, 1024)

        # =========================
        # Expansive Path
        # feature map을 다시 확장하면서 픽셀 단위 복원

        self.dec5_1 = CBR2d(1024, 512)

        # UpSampling (ConvTranspose)
        self.unpool4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        self.dec4_2 = CBR2d(1024, 512)  # skip connection으로 채널 2배
        self.dec4_1 = CBR2d(512, 256)

        self.unpool3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.dec3_2 = CBR2d(512, 256)
        self.dec3_1 = CBR2d(256, 128)

        self.unpool2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        self.dec2_2 = CBR2d(256, 128)
        self.dec2_1 = CBR2d(128, 64)

        self.unpool1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.dec1_2 = CBR2d(128, 64)
        self.dec1_1 = CBR2d(64, 64)

        # Final Layer (1x1 Conv)
        # 각 픽셀에 대해 binary classification 수행
        self.fc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # low-level → high-level feature 추출
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        # Bottleneck
        enc5_1 = self.enc5_1(pool4)


        # 업샘플링 + Skip Connection
        dec5_1 = self.dec5_1(enc5_1)

        # Skip Connection: encoder의 feature map과 결합
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)


        # Output
        # 픽셀 단위 segmentation 결과 생성
        x = self.fc(dec1_1)

        return x