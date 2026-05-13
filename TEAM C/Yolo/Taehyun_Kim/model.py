import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from typing import Optional, List, Tuple, Union


# LocallyConnected2d: 위치마다 독립적인 가중치를 사용하는 레이어
# 일반 Conv2d는 모든 위치에서 같은 가중치(weight sharing)를 사용하지만,
# LocallyConnected2d는 각 위치(patch position)마다 고유한 가중치를 사용한다.
# YOLO v1 논문의 첫 번째 FC 레이어를 대체하는 구조.
class LocallyConnected2d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_h: int,
                 input_w: int,
                 kernel_size: int,
                 stride: Optional[int] = 1,
                 padding: Optional[int] = 0) -> None:
        """
        입력: (N, C, H, W) → 출력: (N, C', H', W')
          H' = floor((H + 2*padding - kernel_size) / stride + 1)
          W' = floor((W + 2*padding - kernel_size) / stride + 1)
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

        # 각 출력 위치(output_h × output_w)마다 별도의 가중치를 가짐 (위치별 독립 필터)
        self.weight = nn.Parameter(th.randn(1, self.in_channels, self.out_channels,
                                            self.output_h, self.output_w,
                                            self.kernel_size, self.kernel_size))
        # 각 (위치, 출력 채널) 쌍마다 독립적인 bias
        self.bias = nn.Parameter(th.randn(1, self.out_channels, self.output_h, self.output_w))

    def forward(self, x: th.Tensor) -> th.Tensor:
        # padding 적용
        x = F.pad(x, (self.padding,) * 4)
        # sliding window 방식으로 패치 추출 (Conv처럼 window를 슬라이딩)
        windows = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)[:, :, None, ...]
        # 각 위치별 가중치와 패치를 element-wise 곱한 후 합산 → bias 추가
        y = th.sum(self.weight * windows, dim=[1, 5, 6]) + self.bias
        return y


# ConvModule: 논문의 Network Design figure에서 한 '블록'에 해당하는 모듈
# 여러 Conv/MaxPool 레이어의 조합을 설정(config) 기반으로 동적 생성
class ConvModule(nn.Module):

    def __init__(self, in_channels: int, module_config: List[Union[List, Tuple]]) -> None:
        """
        module_config 규칙:
          - Conv 레이어:    ('c', kernel_size, out_channels) 또는 ('c', kernel_size, out_channels, stride)
          - MaxPool 레이어: ('p', kernel_size, stride)
          - 반복 블록:      [[layer_1, ..., layer_m], k]  →  layer_1 ~ layer_m을 k번 반복
        """
        super(ConvModule, self).__init__()

        self.layers = []
        for sm_config in module_config:
            if isinstance(sm_config, tuple):
                # 단일 레이어 추가
                in_channels = self._add_layer(in_channels, sm_config)
            elif isinstance(sm_config, list):
                # 반복 블록: [레이어 목록, 반복 횟수]
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
        Conv 레이어: Conv2d → BatchNorm2d → LeakyReLU(0.1)
          - 'same' 패딩 공식: padding = ceil((kernel_size - stride) / 2)
          - BatchNorm이 bias를 흡수하므로 Conv에는 bias=False
          - Kaiming He 초기화 (LeakyReLU에 맞게 설정)
        MaxPool 레이어: MaxPool2d만 단독 사용
        """
        if layer_config[0] == 'c':
            kernel_size, out_channels = layer_config[1:3]
            stride = 1 if len(layer_config) == 3 else layer_config[3]
            # 'same' 패딩으로 feature map 크기를 stride만큼만 줄임
            padding = ceil((kernel_size - stride) / 2)

            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            )
            # Kaiming He 초기화 (LeakyReLU 음수 기울기 a=0.1 반영)
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
        # 레이어들을 순차적으로 통과
        return self.layers(x)



# YOLOv1: 전체 YOLO v1 모델
# - 'classification' 모드: ImageNet 사전학습(pretrain)에 사용
# - 'detection' 모드:      VOC 객체 탐지 학습/추론에 사용
class YOLOv1(nn.Module):

    # ── Backbone 구성 (분류 + 탐지 공통 사용) ──────────────────────────
    # 각 리스트가 하나의 ConvModule (논문의 블록 단위에 대응)
    conv_backbone_config = [
        [('c', 7, 64, 2), ('p', 2, 2)],            # 모듈 1: 7×7 Conv(stride=2) + MaxPool
        [('c', 3, 192), ('p', 2, 2)],               # 모듈 2: 3×3 Conv + MaxPool
        [('c', 1, 128), ('c', 3, 256), ('c', 1, 256), ('c', 3, 512), ('p', 2, 2)],  # 모듈 3
        [[[('c', 1, 256), ('c', 3, 512)], 4], ('c', 1, 512), ('c', 3, 1024), ('p', 2, 2)],  # 모듈 4 (1×3 블록 4회 반복)
        [[[('c', 1, 512), ('c', 3, 1024)], 2]],     # 모듈 5 (1×3 블록 2회 반복)
    ]

    # ── Detection Head 추가 Conv 구성 ─────────────────────────────────
    conv_detection_config = [
        [('c', 3, 1024), ('c', 3, 1024, 2)],        # 3×3 Conv × 2 (마지막은 stride=2)
        [('c', 3, 1024), ('c', 3, 1024)],            # 3×3 Conv × 2
    ]

    def __init__(self, S: int, B: int, C: int, mode: Optional[str] = 'detection') -> None:
        """
        S: 그리드 크기 (이미지를 S×S로 분할)
        B: 각 셀에서 예측하는 바운딩 박스 수
        C: 클래스 수 (VOC=20, ImageNet=1000)
        mode: 'detection' 또는 'classification'
        """
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mode = mode

        # ── Backbone 구성 ─────────────────────────────────────────────
        backbones_modules_list = []
        in_channels = 3  # RGB 입력
        for module_config in YOLOv1.conv_backbone_config:
            cm = ConvModule(in_channels, module_config)
            backbones_modules_list.append(cm)
            in_channels = cm.out_channels
        self.backbone = nn.Sequential(*backbones_modules_list)

        if mode == 'detection':
            # ── Detection Head: 추가 Conv 블록 ──────────────────────────
            head_modules_list = []
            for module_config in YOLOv1.conv_detection_config:
                cm = ConvModule(in_channels, module_config)
                head_modules_list.append(cm)
                in_channels = cm.out_channels
            detection_conv_modules = nn.Sequential(*head_modules_list)

            # ── Detection Head: FC 부분 (LocallyConnected → Flatten → Dropout → Linear) ──
            # 출력 크기: S × S × (C + B*5)
            #   각 셀마다: C개 클래스 확률 + B개 박스 × (objectness + x, y, w, h)
            detection_fc_modules = nn.Sequential(
                LocallyConnected2d(in_channels, 256, 7, 7, 3, 1, 1),  # 위치별 독립 Conv
                nn.LeakyReLU(0.1),
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(256 * 7 * 7, S * S * (C + B * 5))           # 최종 예측 벡터
            )
            # LocallyConnected 레이어 초기화
            nn.init.kaiming_normal_(detection_fc_modules[0].weight, a=0.1, mode='fan_out')
            nn.init.zeros_(detection_fc_modules[0].bias)

            self.detection_head = nn.Sequential(detection_conv_modules, detection_fc_modules)
            self.forward = self._forward_detection

        elif mode == 'classification':
            # ── Classification Head: GlobalAvgPool → Flatten → Linear ──
            self.classification_head = nn.Sequential(
                nn.AvgPool2d(7),       # 7×7 feature map을 1×1로 압축
                nn.Flatten(),
                nn.Linear(1024, C)    # C개 클래스 로짓 출력
            )
            self.forward = self._forward_classification
        else:
            assert -1

    def _forward_classification(self, x: th.Tensor) -> th.Tensor:
        # Backbone → Classification Head → 클래스 로짓
        x = self.backbone(x)
        y = self.classification_head(x)
        return y

    def _forward_detection(self, x: th.Tensor) -> th.Tensor:
        # Backbone → Detection Head → (N, S, S, C+B*5) 형태로 reshape
        x = self.backbone(x)
        x = self.detection_head(x)
        # 1D 예측 벡터를 그리드 형태로 변환
        y = x.reshape(x.shape[0], self.S, self.S, self.C + self.B * 5)
        return y
