"""
model.py 이해용 한국어 주석 버전

핵심 역할:
- YOLOv1 모델 구조 정의
- ImageNet 사전학습용 classification mode 지원
- VOC 객체 탐지용 detection mode 지원

주의:
- 이 파일은 원본 전체 코드 복사본이 아니라 이해용 핵심 구조 재구성입니다.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor


class LocallyConnected2d(nn.Module):
    """
    Locally Connected Layer.

    일반 Conv2d와 비슷하지만 가장 큰 차이는 weight sharing이 없다는 점입니다.

    일반 Conv2d:
    - 같은 필터를 이미지 전체 위치에 반복해서 적용합니다.
    - 위치가 달라도 같은 가중치를 사용합니다.

    LocallyConnected2d:
    - 위치마다 서로 다른 가중치를 사용합니다.
    - 즉, 7x7 feature map의 각 위치가 자기 전용 필터를 가집니다.

    이 저장소는 YOLO 논문 그림의 Fully Connected Layer 대신
    Darknet 구현 방식에 맞춰 Locally Connected Layer를 사용합니다.
    """

    def __init__(self, in_channels, out_channels, input_h, input_w,
                 kernel_size, stride=1, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 출력 feature map의 높이와 너비 계산
        self.output_h = floor((input_h + 2 * padding - kernel_size) / stride + 1)
        self.output_w = floor((input_w + 2 * padding - kernel_size) / stride + 1)

        # 일반 Conv2d라면 weight shape는 보통 다음과 비슷합니다.
        # (out_channels, in_channels, kernel_size, kernel_size)
        #
        # 여기서는 위치마다 가중치가 다르기 때문에 output_h, output_w 차원이 추가됩니다.
        self.weight = nn.Parameter(
            th.randn(
                1,
                in_channels,
                out_channels,
                self.output_h,
                self.output_w,
                kernel_size,
                kernel_size,
            )
        )

        # bias도 output 위치마다 따로 둡니다.
        self.bias = nn.Parameter(th.randn(1, out_channels, self.output_h, self.output_w))

    def forward(self, x):
        # padding 적용
        x = F.pad(x, (self.padding,) * 4)

        # unfold는 이미지를 sliding window 형태로 펼치는 함수입니다.
        # Conv 연산을 직접 구현할 때 자주 쓰입니다.
        windows = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        # weight와 차원을 맞추기 위해 중간에 차원을 하나 추가합니다.
        windows = windows[:, :, None, ...]

        # 각 위치의 window와 각 위치 전용 weight를 곱한 뒤 합산합니다.
        y = th.sum(self.weight * windows, dim=[1, 5, 6]) + self.bias
        return y


class ConvModule(nn.Module):
    """
    YOLOv1의 convolution block을 만드는 클래스입니다.

    원본 코드에서는 네트워크 구조를 튜플 리스트로 정의합니다.

    예시:
    ('c', 3, 192)      -> 3x3 convolution, output channel 192
    ('c', 7, 64, 2)   -> 7x7 convolution, output channel 64, stride 2
    ('p', 2, 2)       -> 2x2 max pooling, stride 2

    반복되는 구조는 [[layer1, layer2], 반복횟수] 형태로 작성합니다.
    """

    def __init__(self, in_channels, module_config):
        super().__init__()
        layers = []

        for item in module_config:
            # 단일 layer 설정
            if isinstance(item, tuple):
                layer, in_channels = self._make_layer(in_channels, item)
                layers.append(layer)

            # 여러 layer를 반복하는 설정
            elif isinstance(item, list):
                repeated_layers, repeat_count = item
                for _ in range(repeat_count):
                    for layer_config in repeated_layers:
                        layer, in_channels = self._make_layer(in_channels, layer_config)
                        layers.append(layer)

        self.out_channels = in_channels
        self.layers = nn.Sequential(*layers)

    def _make_layer(self, in_channels, config):
        if config[0] == 'c':
            # convolution layer
            kernel_size = config[1]
            out_channels = config[2]
            stride = 1 if len(config) == 3 else config[3]

            # same convolution에 가깝게 padding 계산
            padding = ceil((kernel_size - stride) / 2)

            layer = nn.Sequential(
                # BatchNorm을 바로 뒤에 쓰기 때문에 Conv bias는 제거합니다.
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                # YOLOv1/Darknet 계열에서 자주 쓰는 LeakyReLU
                nn.LeakyReLU(0.1),
            )

            # LeakyReLU에 맞춘 He initialization
            nn.init.kaiming_normal_(layer[0].weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            return layer, out_channels

        elif config[0] == 'p':
            # max pooling layer
            kernel_size = config[1]
            stride = config[2]
            return nn.MaxPool2d(kernel_size, stride), in_channels

        else:
            raise ValueError(f"Unknown layer type: {config[0]}")

    def forward(self, x):
        return self.layers(x)


class YOLOv1(nn.Module):
    """
    YOLOv1 전체 모델.

    mode='classification':
        ImageNet 사전학습용 모델입니다.
        backbone 뒤에 average pooling과 linear classifier를 붙입니다.

    mode='detection':
        객체 탐지용 모델입니다.
        backbone 뒤에 detection head를 붙입니다.
        최종 출력은 (N, S, S, C + B*5)입니다.
    """

    def __init__(self, S, B, C, mode='detection'):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mode = mode

        # backbone은 이미지에서 특징을 추출하는 CNN 부분입니다.
        # 원본 코드에서는 conv_backbone_config 리스트로 구조를 정의합니다.
        self.backbone = nn.Sequential(
            # 실제 원본에서는 ConvModule 여러 개가 들어갑니다.
            # 여기서는 핵심 구조 이해를 위해 생략합니다.
        )

        if mode == 'classification':
            # ImageNet 1000개 클래스를 맞히는 사전학습용 head
            self.classification_head = nn.Sequential(
                nn.AvgPool2d(7),
                nn.Flatten(),
                nn.Linear(1024, C),
            )

        elif mode == 'detection':
            # VOC 객체 탐지용 head
            # 원본 코드에서는 추가 ConvModule + LocallyConnected2d + Linear로 구성됩니다.
            self.detection_head = nn.Sequential(
                # detection convolution blocks
                # LocallyConnected2d(...)
                # LeakyReLU
                # Flatten
                # Dropout
                # Linear(256*7*7, S*S*(C+B*5))
            )

        else:
            raise ValueError("mode must be 'classification' or 'detection'")

    def forward(self, x):
        x = self.backbone(x)

        if self.mode == 'classification':
            return self.classification_head(x)

        # detection mode
        x = self.detection_head(x)

        # YOLO 출력 형태로 변환
        # N: batch size
        # S x S: grid
        # C + B*5: 클래스 확률 + B개의 박스 정보
        return x.reshape(x.shape[0], self.S, self.S, self.C + self.B * 5)
