import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 구축하기
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 공통으로 사용될 블록: Conv + BatchNorm + ReLU (CBR2d)
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            # CBR2d 얘는 그냥 임의로 지정한 함수
            layers = []
            # 1. 2D 합성곱 (이미지의 특징 추출)
            # 2D는 단순히 특징을 추출하는 방법 중 하나고, 특징들만 강조된 새로운 맵이 피처맵
            # 2D인 이유는 가로, 세로 두 방향으로 필터가 움직이면서 연산해서
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            # 2. 배치 정규화 (학습 속도 향상 및 안정화)
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            # 3. ReLU 활성화 함수 (비선형성 추가)
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path (Encoder): 이미지 크기를 줄이며 특징을 압축
        # Encoder → 이미지를 작게 압축하면서 이미지 안에 뭐가 있는지 파악하지만, 위치 정보는 많이 손실됨
        # Level 1: 1채널 입력 -> 64채널 출력
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # Level 2: 64채널 -> 128채널
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # Level 3: 128채널 -> 256채널
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # Level 4: 256채널 -> 512채널
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)
        # Bottleneck: 가장 깊은 층 (이미지의 가장 추상적인 정보 추출)
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path (Decoder): 줄어든 이미지를 다시 원래 크기로 복원
        # Decoder → 위치 정보를 여기서 다시 반영 즉, 물체가 어디에 있는지 정확히 그려내고자 함
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)
        # Level 5 Decoder
        # Up-sampling: ConvTranspose2d를 통해 해상도를 2배 키움
        # ConvTranspose2d: 픽셀 사이사이를 메꾸며 크기를 2배로 키우는 '역합성곱'
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        # Skip Connection 결합: Encoder에서 온 데이터(512) + Decoder 데이터(512) = 1024 채널
        # Skip Connection → 인코더가 압축되기 전 가지고 있던 선명한 위치 정보를 디코더에 직접 전달
        # 이 과정을 통해서 아주 정밀한 경계선을 찾아낼 수 있음
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)
        # 최종 출력: 1x1 Conv를 사용해 채널을 1개로 줄임 (세그멘테이션 맵 생성)
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # 데이터의 흐름 (인코더 -> 보틀넥 -> 디코더)
        # Encoder 1~4 과정
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
        # Decoder 5~1 과정 (Skip Connection 중요!)
        dec5_1 = self.dec5_1(enc5_1)
        # torch.cat: 인코더의 특징 맵(enc4_2)을 가져와 디코더(unpool4)와 결합
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

        x = self.fc(dec1_1)

        return x
