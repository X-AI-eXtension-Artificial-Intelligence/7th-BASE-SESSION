"""
model.py - YOLOv1 모델 아키텍처 정의

이 파일은 YOLOv1 논문의 네트워크 구조를 PyTorch로 구현한 것입니다.
주요 구성요소:
  - LocallyConnected2d: 가중치를 공유하지 않는 지역 연결 레이어
  - ConvModule: 논문의 컨볼루션 모듈 (Conv + BN + LeakyReLU 또는 MaxPool)
  - YOLOv1: 전체 모델 (백본 + 탐지/분류 헤드)
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from typing import Optional, List, Tuple, Union


class LocallyConnected2d(nn.Module):
    """
    Locally Connected 2D 레이어.
    일반 Conv2d와 유사하지만, 각 위치(patch)마다 독립적인 가중치를 사용합니다.
    즉, 가중치가 공유되지 않아 위치별로 다른 필터를 학습합니다.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_h: int,
                 input_w: int,
                 kernel_size: int,
                 stride: Optional[int] = 1,
                 padding: Optional[int] = 0) -> None:
        """
        Locally Connected 2D 레이어 초기화.
        입력: (N, C, H, W) → 출력: (N, C', H', W')
          - H' = floor((H + 2*padding - kernel_size) / stride + 1)
          - W' = floor((W + 2*padding - kernel_size) / stride + 1)

        :param in_channels: 입력 채널 수
        :param out_channels: 출력 채널 수 (필터 개수)
        :param input_h: 입력 텐서의 높이 H
        :param input_w: 입력 텐서의 너비 W
        :param kernel_size: 커널 크기 (각 필터: C × kernel_size × kernel_size)
        :param stride: 패치 추출 시 이동 간격
        :param padding: 입력 텐서의 상하좌우에 적용할 패딩 크기
        """
        super(LocallyConnected2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 출력 feature map의 높이/너비 계산
        self.output_h = floor((input_h + 2 * padding - kernel_size) / stride + 1)
        self.output_w = floor((input_w + 2 * padding - kernel_size) / stride + 1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 각 출력 위치마다 독립적인 가중치: (1, in_ch, out_ch, H', W', k, k)
        self.weight = nn.Parameter(th.randn(1, self.in_channels, self.out_channels,
                                            self.output_h, self.output_w,
                                            self.kernel_size, self.kernel_size))

        # 각 (출력 위치, 출력 채널)마다 독립적인 바이어스
        self.bias = nn.Parameter(th.randn(1, self.out_channels, self.output_h, self.output_w))

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        순전파: 각 윈도우 위치에 해당하는 고유 가중치로 곱한 뒤 바이어스를 더합니다.

        :param x: 입력 텐서 (N, C, H, W)
        :return: 출력 텐서 (N, out_channels, H', W')
        """
        # 패딩 적용
        x = F.pad(x, (self.padding,) * 4)
        # unfold로 슬라이딩 윈도우 추출 후 가중치와 element-wise 곱 → 합산
        windows = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)[:, :, None, ...]
        y = th.sum(self.weight * windows, dim=[1, 5, 6]) + self.bias
        return y


class ConvModule(nn.Module):
    """
    YOLOv1 논문의 컨볼루션 모듈 구현.
    Conv + BatchNorm + LeakyReLU 레이어와 MaxPool 레이어의 조합으로 구성됩니다.
    """

    def __init__(self, in_channels: int, module_config: List[Union[List, Tuple]]) -> None:
        """
        모듈 설정(config)에 따라 레이어를 순차적으로 구성합니다.

        설정 형식:
          - Conv 레이어: ('c', kernel_size, out_channels, stride)  (stride 생략 시 1)
          - MaxPool 레이어: ('p', kernel_size, stride)
          - 반복 구조: [[layer1, layer2, ...], 반복횟수]

        패딩은 p = ceil((kernel_size - stride) / 2)로 계산하여 'same' 패딩 효과를 냅니다.

        :param in_channels: 모듈의 입력 채널 수
        :param module_config: 레이어 구성 리스트
        """
        super(ConvModule, self).__init__()

        self.layers = []
        for sm_config in module_config:
            if isinstance(sm_config, tuple):
                # 단일 레이어 추가
                in_channels = self._add_layer(in_channels, sm_config)
            elif isinstance(sm_config, list):
                # 반복 구조: [레이어 리스트, 반복 횟수]
                sm_layers, r = sm_config
                for _ in range(r):
                    for layer_config in sm_layers:
                        in_channels = self._add_layer(in_channels, layer_config)
            else:
                assert -1
        self.out_channels = in_channels
        self.layers = nn.Sequential(*self.layers)

    def _add_layer(self, in_channels: int, layer_config: Tuple) -> int:
        """
        단일 레이어(Conv 또는 MaxPool)를 모듈에 추가합니다.

        Conv 레이어 구성:
          - Conv2d (bias 없음, BN이 상쇄하므로)
          - BatchNorm2d
          - LeakyReLU(0.1)
          - Kaiming 초기화 적용

        :param in_channels: 현재 입력 채널 수
        :param layer_config: 레이어 설정 튜플
        :return: 이 레이어의 출력 채널 수
        """
        if layer_config[0] == 'c':
            kernel_size, out_channels = layer_config[1:3]
            stride = 1 if len(layer_config) == 3 else layer_config[3]
            padding = ceil((kernel_size - stride) / 2)

            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                            bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.LeakyReLU(0.1))
            # Kaiming 초기화: LeakyReLU에 적합한 가중치 초기화
            nn.init.kaiming_normal_(layer[0].weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            self.layers.append(layer)

            in_channels = out_channels

        elif layer_config[0] == 'p':
            kernel_size, stride = layer_config[1:]
            self.layers.append(nn.MaxPool2d(kernel_size, stride))

        else:
            assert -1

        return in_channels

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        입력을 모듈의 레이어들에 순차적으로 통과시킵니다.

        :param x: 입력 텐서
        :return: 출력 텐서
        """
        return self.layers(x)


class YOLOv1(nn.Module):
    """
    YOLOv1 모델.

    동작 모드:
      - 'classification': ImageNet 사전학습용 (백본 + 분류 헤드)
      - 'detection': PASCAL VOC 객체탐지용 (백본 + 탐지 헤드)

    네트워크 구조:
      - conv_backbone_config: 분류/탐지 공통 백본 (24개 Conv 레이어)
      - conv_detection_config: 탐지 전용 추가 Conv 레이어
      - 탐지 헤드: LocallyConnected2d + FC → S×S×(C+B*5) 출력
    """
    # 백본 컨볼루션 모듈 설정 (논문 Figure 3 기반)
    conv_backbone_config = [[('c', 7, 64, 2), ('p', 2, 2)],
                            [('c', 3, 192), ('p', 2, 2)],
                            [('c', 1, 128), ('c', 3, 256), ('c', 1, 256), ('c', 3, 512), ('p', 2, 2)],
                            [[[('c', 1, 256), ('c', 3, 512)], 4], ('c', 1, 512), ('c', 3, 1024), ('p', 2, 2)],
                            [[[('c', 1, 512), ('c', 3, 1024)], 2]]]

    # 탐지 전용 컨볼루션 모듈 설정
    conv_detection_config = [[('c', 3, 1024), ('c', 3, 1024, 2)],
                             [('c', 3, 1024), ('c', 3, 1024)]]

    def __init__(self, S: int, B: int, C: int, mode: Optional[str] = 'detection') -> None:
        """
        YOLOv1 모델 초기화.

        :param S: 그리드 크기 (이미지를 S×S 그리드로 분할)
        :param B: 각 그리드 셀이 예측하는 바운딩 박스 수
        :param C: 클래스 수 (탐지: 20, 분류: 1000)
        :param mode: 'detection' (객체탐지) 또는 'classification' (사전학습)
        """
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mode = mode

        # 백본 네트워크 구성
        backbones_modules_list = []
        in_channels = 3  # RGB 입력
        for module_config in YOLOv1.conv_backbone_config:
            cm = ConvModule(in_channels, module_config)
            backbones_modules_list.append(cm)
            in_channels = cm.out_channels
        self.backbone = nn.Sequential(*backbones_modules_list)

        if mode == 'detection':
            # 탐지 헤드: 추가 Conv 모듈 + LocallyConnected + FC
            head_modules_list = []
            for module_config in YOLOv1.conv_detection_config:
                cm = ConvModule(in_channels, module_config)
                head_modules_list.append(cm)
                in_channels = cm.out_channels
            detection_conv_modules = nn.Sequential(*head_modules_list)
            detection_fc_modules = nn.Sequential(LocallyConnected2d(in_channels, 256, 7, 7, 3, 1, 1),
                                                 nn.LeakyReLU(0.1),
                                                 nn.Flatten(),
                                                 nn.Dropout(p=0.5),  # 과적합 방지
                                                 nn.Linear(256 * 7 * 7, S * S * (C + B * 5)))

            nn.init.kaiming_normal_(detection_fc_modules[0].weight, a=0.1, mode='fan_out')
            nn.init.zeros_(detection_fc_modules[0].bias)

            self.detection_head = nn.Sequential(detection_conv_modules,
                                                detection_fc_modules)
            self.forward = self._forward_detection

        elif mode == 'classification':
            # 분류 헤드: AvgPool + FC (ImageNet 사전학습용)
            self.classification_head = nn.Sequential(nn.AvgPool2d(7),
                                                     nn.Flatten(),
                                                     nn.Linear(1024, C))
            self.forward = self._forward_classification

        else:
            assert -1

    def _forward_classification(self, x: th.Tensor) -> th.Tensor:
        """
        분류 모드 순전파: 백본 → 분류 헤드

        :param x: 입력 이미지 텐서 (N, 3, 224, 224)
        :return: 클래스 로짓 (N, C)
        """
        x = self.backbone(x)
        y = self.classification_head(x)
        return y

    def _forward_detection(self, x: th.Tensor) -> th.Tensor:
        """
        탐지 모드 순전파: 백본 → 탐지 헤드 → (N, S, S, C+B*5) 형태로 reshape

        :param x: 입력 이미지 텐서 (N, 3, 448, 448)
        :return: 예측 텐서 (N, S, S, C+B*5)
                 각 셀: [클래스 확률 C개 | 신뢰도1, x, y, w, h | 신뢰도2, x, y, w, h]
        """
        x = self.backbone(x)
        x = self.detection_head(x)
        y = x.reshape(x.shape[0], self.S, self.S, self.C + self.B * 5)
        return y
