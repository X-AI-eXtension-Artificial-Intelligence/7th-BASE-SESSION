import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from typing import Optional, List, Tuple, Union


class LocallyConnected2d(nn.Module):
    """
    위치마다 독립된 가중치를 사용하는 2D Locally Connected Layer.

    일반 Conv2d와의 차이:
      - Conv2d      : 모든 위치에서 동일한 필터 가중치 공유 (weight sharing)
      - LocallyConn : 각 출력 위치 (h', w')마다 별도의 필터 가중치 보유

    [가중치 형태]
      weight : (1, in_channels, out_channels, H', W', kH, kW)
               → 총 파라미터 수 = in_channels * out_channels * H' * W' * kH * kW
               (Conv2d 대비 H'*W' 배 더 많은 파라미터)

    [용도]
      YOLOv1 detection head의 마지막 공간적 처리 레이어로 사용.
      그리드 셀마다 서로 다른 수용 영역 특성을 학습할 수 있게 해줌.

    [제약]
      - 출력 해상도가 고정되어 있어 가중치 크기가 정해짐.
        따라서 학습 시와 다른 입력 크기는 사용 불가.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_h: int,
                 input_w: int,
                 kernel_size: int,
                 stride: Optional[int] = 1,
                 padding: Optional[int] = 0) -> None:
        super(LocallyConnected2d, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        # 출력 해상도: Conv2d와 동일 공식
        self.output_h = floor((input_h + 2 * padding - kernel_size) / stride + 1)
        self.output_w = floor((input_w + 2 * padding - kernel_size) / stride + 1)
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding

        # 위치별 독립 가중치: (1, C_in, C_out, H', W', kH, kW)
        # 1 차원은 배치 차원 브로드캐스팅용
        self.weight = nn.Parameter(th.randn(1, self.in_channels, self.out_channels,
                                            self.output_h, self.output_w,
                                            self.kernel_size, self.kernel_size))

        # 위치 × 채널별 독립 바이어스: (1, C_out, H', W')
        self.bias = nn.Parameter(th.randn(1, self.out_channels, self.output_h, self.output_w))

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        [연산 흐름]
        1. 패딩 적용
        2. unfold 로 sliding window 추출 → (N, C_in, H', W', kH, kW)
        3. 위치별 가중치와 element-wise 곱 후 C_in, kH, kW 차원 sum
        4. 바이어스 덧셈

        [unfold 동작]
          x.unfold(dim, size, step) : dim 방향으로 size 크기의 윈도우를
          step 간격으로 슬라이딩하여 새 차원으로 펼침.

        [브로드캐스팅]
          weight: (1, C_in, C_out, H', W', kH, kW)
          windows:(N, C_in, 1,    H', W', kH, kW)  ← None으로 C_out 차원 삽입
          sum 이후 결과: (N, C_out, H', W')

        :param x: (N, C_in, H, W) 입력 텐서
        :return:  (N, C_out, H', W') 출력 텐서
        """
        x = F.pad(x, (self.padding,) * 4)  # 상하좌우 동일 패딩
        # windows : (N, C_in, H', W', kH, kW) → None으로 C_out 차원 추가
        windows = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)[:, :, None, ...]
        # dim=[1,5,6] : C_in, kH, kW 차원을 합산 → (N, C_out, H', W')
        y = th.sum(self.weight * windows, dim=[1, 5, 6]) + self.bias
        return y


class ConvModule(nn.Module):
    """
    설정 리스트로부터 Conv/MaxPool 레이어 시퀀스를 동적으로 생성하는 모듈.

    [설정 포맷]
      - 합성곱 레이어 : ('c', kernel_size, out_channels)
                        ('c', kernel_size, out_channels, stride)  ← stride 있는 경우
      - 풀링 레이어   : ('p', kernel_size, stride)
      - 반복 블록     : [[layer1, layer2, ...], repeat_count]

    [패딩 전략]
      stride가 없거나 stride=1일 때 'same' 패딩 유지:
        padding = ceil((kernel_size - stride) / 2)
      → 공간 해상도는 stride에 의해서만 줄어듦.

    [Conv 레이어 구성]
      Conv2d (bias=False) → BatchNorm2d → LeakyReLU(0.1)
      - bias 제거: BN이 평균을 정규화하므로 bias 항은 무의미 (메모리 절약)
      - Kaiming 초기화: LeakyReLU와 함께 쓸 때 적합한 초기화 방법
    """

    def __init__(self, in_channels: int, module_config: List[Union[List, Tuple]]) -> None:
        """
        :param in_channels:   첫 번째 레이어의 입력 채널 수
        :param module_config: 레이어 설정 리스트 (위 포맷 참조)
        """
        super(ConvModule, self).__init__()

        self.layers = []
        for sm_config in module_config:
            if isinstance(sm_config, tuple):
                # 단일 레이어
                in_channels = self._add_layer(in_channels, sm_config)
            elif isinstance(sm_config, list):
                # [레이어 목록, 반복 횟수] 형태
                sm_layers, r = sm_config
                for _ in range(r):
                    for layer_config in sm_layers:
                        in_channels = self._add_layer(in_channels, layer_config)
            else:
                assert -1  # 알 수 없는 설정 포맷
        self.out_channels = in_channels  # 다음 모듈에 전달할 채널 수
        self.layers = nn.Sequential(*self.layers)

    def _add_layer(self, in_channels: int, layer_config: Tuple) -> int:
        """
        단일 레이어(Conv 또는 MaxPool)를 self.layers에 추가.

        [Conv 구성 상세]
          - padding = ceil((k - s) / 2) : stride만큼 해상도를 줄이는 'same-like' 패딩
          - Kaiming normal init (fan_out 모드, leaky_relu nonlinearity)

        [MaxPool]
          - 별도의 활성화 함수나 BN 없이 풀링만 수행

        :param in_channels:  현재 레이어의 입력 채널
        :param layer_config: ('c', ...) 또는 ('p', ...) 형태의 튜플
        :return: 현재 레이어의 출력 채널 수
        """
        if layer_config[0] == 'c':
            kernel_size, out_channels = layer_config[1:3]
            stride  = 1 if len(layer_config) == 3 else layer_config[3]
            padding = ceil((kernel_size - stride) / 2)  # same-like padding

            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            )
            # Kaiming 초기화 (fan_out: 정방향 신호 분산 유지)
            nn.init.kaiming_normal_(layer[0].weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            self.layers.append(layer)
            in_channels = out_channels

        elif layer_config[0] == 'p':
            kernel_size, stride = layer_config[1:]
            self.layers.append(nn.MaxPool2d(kernel_size, stride))

        else:
            assert -1  # 지원하지 않는 레이어 타입

        return in_channels

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        입력을 레이어 시퀀스에 순차적으로 통과시킴.

        :param x: 입력 텐서
        :return: 출력 텐서
        """
        return self.layers(x)


class YOLOv1(nn.Module):
    """
    [아키텍처 개요]
    Backbone (공유):
      conv1: 7×7 Conv, stride=2 → MaxPool
      conv2: 3×3 Conv           → MaxPool
      conv3: 1×1+3×3 블록       → MaxPool
      conv4: 1×1+3×3 반복 블록  → MaxPool
      conv5: 1×1+3×3 반복 블록  (backbone 마지막)

    Detection Head (파인튜닝 전용):
      conv6: 3×3 Conv × 2 (stride=2 포함)
      conv7: 3×3 Conv × 2
      LC2d : Locally Connected (7×7, 256ch)
      FC   : 256×7×7 → S×S×(C+B×5)

    Classification Head (사전학습 전용):
      AvgPool(7) → Flatten → Linear(1024, C)

    [주요 설계 결정]
      - backbone의 마지막 모듈(conv5)과 detection head의 첫 모듈(conv6)이
        논문 Figure에서는 하나의 모듈로 표현되어 있으나, 코드에서는 분리하여 관리.
      - LocallyConnected2d 후 Dropout(p=0.5) 적용 → 과적합 방지
      - 출력 레이어에 Sigmoid/Softmax 없음 → raw logit 형태로 출력
        (손실 함수에서 MSE 직접 적용)
    """

    # Backbone 설정: 분류/탐지 공통
    conv_backbone_config = [
        [('c', 7, 64, 2), ('p', 2, 2)],                          # conv1
        [('c', 3, 192), ('p', 2, 2)],                             # conv2
        [('c', 1, 128), ('c', 3, 256), ('c', 1, 256), ('c', 3, 512), ('p', 2, 2)],  # conv3
        [[[('c', 1, 256), ('c', 3, 512)], 4], ('c', 1, 512), ('c', 3, 1024), ('p', 2, 2)],  # conv4
        [[[('c', 1, 512), ('c', 3, 1024)], 2]]                    # conv5
    ]

    # Detection Head 설정: 탐지 전용
    conv_detection_config = [
        [('c', 3, 1024), ('c', 3, 1024, 2)],                      # conv6 (stride=2로 7×7 생성)
        [('c', 3, 1024), ('c', 3, 1024)]                           # conv7
    ]

    def __init__(self, S: int, B: int, C: int, mode: Optional[str] = 'detection') -> None:
        """
        :param S:    그리드 분할 수 (S×S, 보통 7)
        :param B:    셀당 예측 박스 수 (보통 2)
        :param C:    클래스 수 (PASCAL VOC = 20)
        :param mode: 'detection' 또는 'classification'
        """
        super(YOLOv1, self).__init__()
        self.S    = S
        self.B    = B
        self.C    = C
        self.mode = mode

        # Backbone 빌드
        backbones_modules_list = []
        in_channels = 3  # RGB 입력
        for module_config in YOLOv1.conv_backbone_config:
            cm = ConvModule(in_channels, module_config)
            backbones_modules_list.append(cm)
            in_channels = cm.out_channels
        self.backbone = nn.Sequential(*backbones_modules_list)
        # backbone 출력: (N, 1024, 14, 14) for D=448

        # Head 빌드 
        if mode == 'detection':
            head_modules_list = []
            for module_config in YOLOv1.conv_detection_config:
                cm = ConvModule(in_channels, module_config)
                head_modules_list.append(cm)
                in_channels = cm.out_channels
            detection_conv_modules = nn.Sequential(*head_modules_list)
            # conv head 출력: (N, 1024, 7, 7)

            detection_fc_modules = nn.Sequential(
                LocallyConnected2d(in_channels, 256, 7, 7, 3, 1, 1),
                # ^ (N, 256, 7, 7) — 위치별 독립 가중치
                nn.LeakyReLU(0.1),
                nn.Flatten(),
                nn.Dropout(p=0.5),   # 과적합 억제
                nn.Linear(256 * 7 * 7, S * S * (C + B * 5))
                # ^ 최종 출력: S*S*(C+B*5) 개의 raw logit
            )

            # LocallyConnected2d 초기화
            nn.init.kaiming_normal_(detection_fc_modules[0].weight, a=0.1, mode='fan_out')
            nn.init.zeros_(detection_fc_modules[0].bias)

            self.detection_head = nn.Sequential(detection_conv_modules, detection_fc_modules)
            self.forward = self._forward_detection

        elif mode == 'classification':
            # 사전학습: backbone → GlobalAvgPool → Linear(C)
            self.classification_head = nn.Sequential(
                nn.AvgPool2d(7),   # (N, 1024, 14, 14) → (N, 1024, 1, 1) [D=448 기준 14×14]
                nn.Flatten(),
                nn.Linear(1024, C)
            )
            self.forward = self._forward_classification

        else:
            assert -1  # 지원하지 않는 모드

    def _forward_classification(self, x: th.Tensor) -> th.Tensor:
        """
        사전학습(분류) 추론 경로.

        backbone → classification_head → (N, C) 클래스 로짓

        :param x: (N, 3, D, D) 입력 이미지
        :return:  (N, C) 클래스 로짓
        """
        x = self.backbone(x)
        y = self.classification_head(x)
        return y

    def _forward_detection(self, x: th.Tensor) -> th.Tensor:
        """
        탐지(파인튜닝) 추론 경로.

        backbone → detection_head → reshape → (N, S, S, C+B*5)

        출력 텐서 구조 (마지막 차원):
          [cls_0, ..., cls_{C-1},          ← C개 클래스 확률
           conf_0, cx_0, cy_0, sw_0, sh_0, ← 박스 0
           conf_1, cx_1, cy_1, sw_1, sh_1  ← 박스 1  (B=2 기준)
          ]

        :param x: (N, 3, D, D) 입력 이미지
        :return:  (N, S, S, C+B*5) 그리드별 예측값
        """
        x = self.backbone(x)
        x = self.detection_head(x)
        y = x.reshape(x.shape[0], self.S, self.S, self.C + self.B * 5)
        return y