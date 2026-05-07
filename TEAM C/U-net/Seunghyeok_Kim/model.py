import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 구축하기
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):            # Block 구성 : Conv -> BatchNorm -> Activation
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,                    # Default 파라미터 구성 [ 입력 형상, 출력 형상, 필터 크기, Stride, Padding, 편향 ]
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]                                       # 정규화 함수 정의 Batch Normalization
            layers += [nn.ReLU()]                                                                       # 활성화 함수 정의 ReLU

            cbr = nn.Sequential(*layers)                                                                # Layer 파라미터 Sequential 정의

            return cbr

        # Encoder -> Feature Map 크기 감소 / 차원 증가 : Encoding 진행
        
        # Contracting path
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)             # In Channel 1 -> Out Channel 64 : 차원 증가 
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)            # In Channel 64 -> Out Channel 64 : 비선형성 증가

        self.pool1 = nn.MaxPool2d(kernel_size=2)                        # Max Pooling : Size 2 Downsampling(1/2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)           # In Channel 64 -> Out Channel 128 : 차원 증가 
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)          # In Channel 128 -> Out Channel 128 : 비선형성 증가

        self.pool2 = nn.MaxPool2d(kernel_size=2)                        # Max Pooling : Size 2 Downsampling(1/2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)          # In Channel 128 -> Out Channel 256 : 차원 증가 
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)          # In Channel 256 -> Out Channel 256 : 비선형성 증가

        self.pool3 = nn.MaxPool2d(kernel_size=2)                        # Max Pooling : Size 2 Downsampling(1/2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)          # In Channel 256 -> Out Channel 512 : 차원 증가 
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)          # In Channel 512 -> Out Channel 512 : 비선형성 증가

        self.pool4 = nn.MaxPool2d(kernel_size=2)                        # Max Pooling : Size 2 Downsampling(1/2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)         # In Channel 512 -> Out Channel 1024 : 차원 증가 

        # Decoder -> Feature Map 크기 증가 / 차원 감소 : Decoding 진행
        
        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)                             # In Channel 1024 -> Out Channel 512 : 차원 감소

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,                # In Channel 512 -> Out Channel 512 : Upsampling(2)
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)                          # In Channel 1024 -> Out Channel 512 : 차원 감소
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)                              # In Channel 512 -> Out Channel 256 : 차원 감소

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,                # In Channel 256 -> Out Channel 256 : Upsampling(2)
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)                          # In Channel 256 + 256(Encoder 잔차 연결) -> Out Channel 256 : 차원 감소
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)                              # In Channel 256 -> Out Channel 128 : 차원 감소

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,                # In Channel 128 -> Out Channel 128 : Upsampling(2)
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)                          # In Channel 128 + 128(Encoder 잔차 연결) -> Out Channel 128 : 차원 감소
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)                               # In Channel 128 -> Out Channel 64 : 차원 감소

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,                  # In Channel 64 -> Out Channel 64 : Upsampling(2)
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)                            # In Channel 64 + 64(Encoder 잔차 연결) -> Out Channel 64 : 차원 감소
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)                                # In Channel 64 -> Out Channel 64 : 비선형성 증가

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)     # Fully Connected Layer

    def forward(self, x):                            # 순전파 연산 시행
        enc1_1 = self.enc1_1(x)                      # Conv 1
        enc1_2 = self.enc1_2(enc1_1)                 # Conv 2
        pool1 = self.pool1(enc1_2)                   # DownSampling

        enc2_1 = self.enc2_1(pool1)                  # Conv 1
        enc2_2 = self.enc2_2(enc2_1)                 # Conv 2
        pool2 = self.pool2(enc2_2)                   # DownSampling

        enc3_1 = self.enc3_1(pool2)                  # Conv 1
        enc3_2 = self.enc3_2(enc3_1)                 # Conv 2
        pool3 = self.pool3(enc3_2)                   # DownSampling

        enc4_1 = self.enc4_1(pool3)                  # Conv 1
        enc4_2 = self.enc4_2(enc4_1)                 # Conv 2
        pool4 = self.pool4(enc4_2)                   # DownSampling

        enc5_1 = self.enc5_1(pool4)                  # 차원 증가

        dec5_1 = self.dec5_1(enc5_1)                 # 차원 감소

        unpool4 = self.unpool4(dec5_1)               # UpSampling
        cat4 = torch.cat((unpool4, enc4_2), dim=1)   # Concat(Encoder Out + Decoder)
        dec4_2 = self.dec4_2(cat4)                   # 차원 감소
        dec4_1 = self.dec4_1(dec4_2)                 # 차원 감소

        unpool3 = self.unpool3(dec4_1)               # UpSampling
        cat3 = torch.cat((unpool3, enc3_2), dim=1)   # Concat(Encoder Out + Decoder)
        dec3_2 = self.dec3_2(cat3)                   # 차원 감소
        dec3_1 = self.dec3_1(dec3_2)                 # 차원 감소

        unpool2 = self.unpool2(dec3_1)               # UpSampling
        cat2 = torch.cat((unpool2, enc2_2), dim=1)   # Concat(Encoder Out + Decoder)
        dec2_2 = self.dec2_2(cat2)                   # 차원 감소
        dec2_1 = self.dec2_1(dec2_2)                 # 차원 감소

        unpool1 = self.unpool1(dec2_1)               # UpSampling
        cat1 = torch.cat((unpool1, enc1_2), dim=1)   # Concat(Encoder Out + Decoder)
        dec1_2 = self.dec1_2(cat1)                   # 차원 감소
        dec1_1 = self.dec1_1(dec1_2)                 # 비선형 증가

        x = self.fc(dec1_1)                          # Fully Connected Layer

        return x                                     # 최종 Output 출력
