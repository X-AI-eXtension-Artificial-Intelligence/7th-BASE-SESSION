import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from typing import Optional, List, Tuple, Union

# 1. Locally Connected 2D Layer 정의
# 일반적인 Convolution과 달리 가중치(Weights)를 공유하지 않고, 각 위치마다 독립적인 가중치를 사용
class LocallyConnected2d(nn.Module):
    """
    Locally Connected 2D Layer는 2D Convolution과 유사하게 동작하지만,
    가장 큰 차이점은 가중치가 공유되지 않는다는 것이다. 대신, 각 윈도우 위치마다 고유한 가중치 세트를 가진다.
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
        Locally Connected 2D Layer 초기화.
        입력 텐서 차원: (N, C, H, W) -> 출력 텐서 차원: (N, C', H', W')
        """
        super(LocallyConnected2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 출력 피처맵의 높이 H' 계산
        self.output_h = floor((input_h + 2 * padding - kernel_size) / stride + 1)
        # 출력 피처맵의 너비 W' 계산
        self.output_w = floor((input_w + 2 * padding - kernel_size) / stride + 1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 가중치 파라미터 생성: (1, 입력채널, 출력채널, 출력높이, 출력너비, 커널높이, 커널너비)
        # 위치(output_h, output_w)마다 별도의 커널 가중치를 가진다는 점이 핵심
        self.weight = nn.Parameter(th.randn(1, self.in_channels, self.out_channels,
                                            self.output_h, self.output_w,
                                            self.kernel_size, self.kernel_size))

        # 바이어스 파라미터 생성: (1, 출력채널, 출력높이, 출력너비)
        self.bias = nn.Parameter(th.randn(1, self.out_channels, self.output_h, self.output_w))

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        순전파 단계. Convolution처럼 윈도우를 추출하지만, 각 위치마다 다른 가중치를 곱함
        """
        # 입력 데이터 가장자리에 패딩 추가
        x = F.pad(x, (self.padding,) * 4)
        # unfold를 사용하여 슬라이딩 윈도우 방식으로 패치(Patch)를 추출
        # 차원 조작을 통해 (N, C, H_out, W_out, k, k) 형태의 윈도우 텐서를 만든다.
        windows = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)[:, :, None, ...]
        # 추출된 윈도우와 위치별 가중치를 곱하고 채널/커널 차원에 대해 합산한 뒤 바이어스를 더함
        y = th.sum(self.weight * windows, dim=[1, 5, 6]) + self.bias
        return y

# 2. ConvModule 정의
# 논문의 Network Design 그림에 등장하는 반복적인 컨볼루션 블록들을 구성하기 위한 클래스
class ConvModule(nn.Module):
    """
    ConvModule은 논문에 제시된 아키텍처 설정을 기반으로 컨볼루션 레이어들을 순차적으로 생성
    """

    def __init__(self, in_channels: int, module_config: List[Union[List, Tuple]]) -> None:
        """
        설정 리스트를 받아 레이어를 쌓음
        - ('c', ...): 컨볼루션 레이어
        - ('p', ...): 맥스풀링 레이어
        - [[layer_configs], k]: 특정 레이어 조합을 k번 반복
        """
        super(ConvModule, self).__init__()

        self.layers = []
        # 설정값들을 순회하며 레이어 추가
        for sm_config in module_config:
            if isinstance(sm_config, tuple):
                # 튜플 형태면 단일 레이어 추가
                in_channels = self._add_layer(in_channels, sm_config)
            elif isinstance(sm_config, list):
                # 리스트 형태면 [레이어들, 반복횟수]로 해석하여 반복 추가
                sm_layers, r = sm_config
                for _ in range(r):
                    for layer_config in sm_layers:
                        in_channels = self._add_layer(in_channels, layer_config)
            else:
                assert -1
        # 최종 출력 채널 수 저장
        self.out_channels = in_channels
        # 리스트에 담긴 레이어들을 nn.Sequential로 묶음
        self.layers = nn.Sequential(*self.layers)

    def _add_layer(self, in_channels: int, layer_config: Tuple) -> int:
        """
        컨볼루션 또는 맥스풀링 레이어를 실제로 생성하여 리스트에 추가
        컨볼루션은 Conv2d -> BatchNorm -> LeakyReLU 순서로 구성
        """
        # 'c'로 시작하면 컨볼루션 레이어 생성
        if layer_config[0] == 'c':
            kernel_size, out_channels = layer_config[1:3]
            # 스트라이드 값이 없으면 기본값 1 사용
            stride = 1 if len(layer_config) == 3 else layer_config[3]
            # 입력과 출력의 가로세로 크기를 유지하기 위한 'Same' 패딩 계산
            padding = ceil((kernel_size - stride) / 2)

            # 레이어 시퀀스 정의: BatchNorm을 쓰기 때문에 Conv2d의 bias는 False 설정
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                            bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.LeakyReLU(0.1))
            # LeakyReLU 활성화 함수에 적합한 Kaiming He 초기화 적용
            nn.init.kaiming_normal_(layer[0].weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            self.layers.append(layer)

            # 다음 레이어를 위해 출력 채널을 입력 채널로 업데이트
            in_channels = out_channels

        # 'p'로 시작하면 맥스풀링 레이어 생성
        elif layer_config[0] == 'p'
            kernel_size, stride = layer_config[1:]
            self.layers.append(nn.MaxPool2d(kernel_size, stride))

        else:
            assert -1

        return in_channels

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        입력 데이터를 시퀀셜 레이어에 통과시킴
        """
        return self.layers(x)


# 3. YOLOv1 메인 모델 클래스 정의
# 모드에 따라 ImageNet 사전 학습용(Classification) 또는 VOC 탐지용(Detection) 구조를 가짐
class YOLOv1(nn.Module):
    """
    YOLOv1 모델. ImageNet으로 사전 학습된 후 PASCAL VOC 데이터로 파인튜닝
    """
    # 논문의 Table 1 구조를 그대로 리스트로 정의한 백본(Backbone) 설정
    conv_backbone_config = [[('c', 7, 64, 2), ('p', 2, 2)]
                            [('c', 3, 192), ('p', 2, 2)],
                            [('c', 1, 128), ('c', 3, 256), ('c', 1, 256), ('c', 3, 512), ('p', 2, 2)],
                            [[[('c', 1, 256), ('c', 3, 512)], 4], ('c', 1, 512), ('c', 3, 1024), ('p', 2, 2)],
                            [[[('c', 1, 512), ('c', 3, 1024)], 2]]]

    # 탐지(Detection) 태스크에서만 추가로 사용하는 레이어 설정
    conv_detection_config = [[('c', 3, 1024), ('c', 3, 1024, 2)],
                             [('c', 3, 1024), ('c', 3, 1024)]]

    def __init__(self, S: int, B: int, C: int, mode: Optional[str] = 'detection') -> None:
        """
        YOLO 모델 초기화. 모드(detection/classification)에 따라 아키텍처가 바뀌게 됨
        S: 격자 크기(7x7), B: 격자당 상자 수(2), C: 클래스 수(VOC=20, ImageNet=1000)
        """
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mode = mode

        # 1. 공통 백본(Backbone) 네트워크 생성
        backbones_modules_list = []
        in_channels = 3
        for module_config in YOLOv1.conv_backbone_config:
            cm = ConvModule(in_channels, module_config)
            backbones_modules_list.append(cm)
            in_channels = cm.out_channels
        self.backbone = nn.Sequential(*backbones_modules_list)

        # 2. 모드에 따른 헤드(Head) 구성
        if mode == 'detection':
            # 탐지 모드: 추가 컨볼루션 레이어 생성
            head_modules_list = []
            for module_config in YOLOv1.conv_detection_config:
                cm = ConvModule(in_channels, module_config)
                head_modules_list.append(cm)
                in_channels = cm.out_channels
            detection_conv_modules = nn.Sequential(*head_modules_list)
            
            # 탐지용 전결합(FC) 모듈 정의: 마지막에 LocallyConnected2d 레이어가 포함됨
            detection_fc_modules = nn.Sequential(LocallyConnected2d(in_channels, 256, 7, 7, 3, 1, 1),
                                                 nn.LeakyReLU(0.1),
                                                 nn.Flatten(), # 텐서를 1차원으로 펼침
                                                 nn.Dropout(p=0.5), # 과적합 방지용 드롭아웃
                                                 # 최종 출력: (7 * 7 * (20 + 2 * 5)) = 1470차원 벡터
                                                 nn.Linear(256 * 7 * 7, S * S * (C + B * 5)))

            # FC 레이어 가중치 초기화
            nn.init.kaiming_normal_(detection_fc_modules[0].weight, a=0.1, mode='fan_out')
            nn.init.zeros_(detection_fc_modules[0].bias)

            # 컨볼루션과 FC를 묶어 탐지 헤드 완성
            self.detection_head = nn.Sequential(detection_conv_modules,
                                                detection_fc_modules)
            # 순전파 함수를 탐지용으로 설정
            self.forward = self._forward_detection

        elif mode == 'classification':
            # 사전 학습 모드: GAP(Global Average Pooling)와 선형 레이어로 구성
            self.classification_head = nn.Sequential(nn.AvgPool2d(7),
                                                     nn.Flatten(),
                                                     nn.Linear(1024, C))
            # 순전파 함수를 분류용으로 설정
            self.forward = self._forward_classification

        else:
            assert -1

    def _forward_classification(self, x: th.Tensor) -> th.Tensor:
        """
        분류(Classification)를 위한 순전파. 이미지넷 사전 학습 시 사용
        """
        x = self.backbone(x)
        y = self.classification_head(x)
        return y

    def _forward_detection(self, x: th.Tensor) -> th.Tensor:
        """
        물체 탐지(Detection)를 위한 순전파. VOC 학습 및 테스트 시 사용
        """
        # 1. 백본을 통해 이미지 특징 추출
        x = self.backbone(x)
        # 2. 탐지 헤드를 통해 좌표 및 확률 예측
        x = self.detection_head(x)
        # 3. 1차원 벡터를 다시 (Batch, 7, 7, 30) 격자 구조로 리쉐이프(Reshape)
        # 30차원 = 20(클래스 확률) + 2(상자별 신뢰도) + 8(상자별 x, y, w, h 좌표)
        y = x.reshape(x.shape[0], self.S, self.S, self.C + self.B * 5)
        return y
