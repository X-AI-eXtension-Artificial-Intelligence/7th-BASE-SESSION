import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from typing import Optional, List, Tuple, Union


# ==============================================================================
# LocallyConnected2d
# ==============================================================================

class LocallyConnected2d(nn.Module):
    """
    [일반 Conv2d와의 차이]
    Conv2d는 모든 위치에서 동일한 필터(가중치 공유)를 사용한다.
    LocallyConnected2d는 위치마다 고유한 가중치를 가진다.

        Conv2d:            weight shape = (out_ch, in_ch, kH, kW)
        LocallyConnected:  weight shape = (in_ch, out_ch, H', W', kH, kW)
                           → H'×W' 각 위치마다 별도의 필터가 존재

    논문에서는 마지막 FC 레이어 대신 이 레이어를 사용해
    7×7 그리드의 각 위치가 독립적인 특징을 학습하도록 한다.
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
        :param in_channels:  입력 채널 수
        :param out_channels: 출력 채널 수 (필터 수)
        :param input_h:      입력 텐서의 높이 H
        :param input_w:      입력 텐서의 너비 W
        :param kernel_size:  커널 크기 (정사각형)
        :param stride:       슬라이딩 보폭
        :param padding:      패딩 크기
        """
        super(LocallyConnected2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 출력 공간 크기 계산 (Conv2d와 동일한 공식)
        self.output_h = floor((input_h + 2 * padding - kernel_size) / stride + 1)
        self.output_w = floor((input_w + 2 * padding - kernel_size) / stride + 1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # [핵심] 위치마다 고유한 가중치: (1, in_ch, out_ch, H', W', kH, kW)
        # Conv2d였다면 (out_ch, in_ch, kH, kW)로 위치 정보가 없음
        self.weight = nn.Parameter(th.randn(1, self.in_channels, self.out_channels,
                                            self.output_h, self.output_w,
                                            self.kernel_size, self.kernel_size))

        # 편향도 위치마다 다름: (1, out_ch, H', W')
        self.bias = nn.Parameter(th.randn(1, self.out_channels, self.output_h, self.output_w))

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Conv2d와 같은 방식으로 윈도우를 추출하되,
        각 위치의 윈도우에 해당 위치만의 가중치를 곱한다.

        처리 흐름:
          1. 패딩 적용
          2. unfold로 모든 윈도우 추출: shape (N, in_ch, H', W', kH, kW)
          3. None 삽입으로 out_ch 차원 추가: (N, in_ch, 1, H', W', kH, kW)
          4. weight (1, in_ch, out_ch, H', W', kH, kW)와 브로드캐스트 곱셈
          5. in_ch, kH, kW 차원을 합산 → (N, out_ch, H', W')
          6. 위치별 bias 추가
        """
        x = F.pad(x, (self.padding,) * 4)

        # unfold: 슬라이딩 윈도우 패치 추출
        # shape: (N, in_ch, H', W', kH, kW) → None 삽입 → (N, in_ch, 1, H', W', kH, kW)
        windows = x.unfold(2, self.kernel_size, self.stride) \
                   .unfold(3, self.kernel_size, self.stride)[:, :, None, ...]

        # 위치별 가중치와 곱한 후 in_ch(dim=1), kH(dim=5), kW(dim=6) 합산
        y = th.sum(self.weight * windows, dim=[1, 5, 6]) + self.bias
        return y


# ==============================================================================
# ConvModule
# ==============================================================================

class ConvModule(nn.Module):
    """
    논문 Figure 3의 각 컨볼루션 블록을 설정(config)으로 정의하는 모듈.

    설정 형식:
      - 합성곱 레이어: ('c', kernel_size, out_channels)
                   또는 ('c', kernel_size, out_channels, stride)
      - 맥스풀 레이어: ('p', kernel_size, stride)
      - 반복 블록:    [[레이어1, 레이어2, ...], 반복횟수]

    예시:
      [[[('c', 1, 256), ('c', 3, 512)], 4], ('c', 1, 512), ('c', 3, 1024), ('p', 2, 2)]
      → 1×1 conv + 3×3 conv를 4번 반복 후, 1×1 conv, 3×3 conv, MaxPool

    [YOLO의 특징] 1×1 축소 레이어
    GoogLeNet의 Inception 모듈 대신 1×1 conv를 사용해 채널 수를 줄인다.
    → 연산량을 줄이면서 비선형성을 추가하는 Lin et al. 방식
    """

    def __init__(self, in_channels: int, module_config: List[Union[List, Tuple]]) -> None:
        super(ConvModule, self).__init__()

        self.layers = []
        for sm_config in module_config:
            if isinstance(sm_config, tuple):
                # 단일 레이어
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
        레이어 한 개를 생성하여 self.layers에 추가.

        합성곱 레이어 구성:
          Conv2d (bias=False) → BatchNorm2d → LeakyReLU(0.1)

          [bias=False인 이유]
          BatchNorm이 평균을 0으로 정규화하므로 bias를 더해도 상쇄된다.
          불필요한 파라미터를 제거해 학습 효율을 높인다.

          [패딩 공식: ceil((kernel_size - stride) / 2)]
          stride=1이면 입력과 출력의 H, W가 동일하게 유지된다 (same padding).
          stride=2이면 공간 크기가 절반으로 줄어든다.

          [Leaky ReLU (α=0.1)]
          일반 ReLU는 음수 입력에서 기울기가 0 → 뉴런이 죽는 문제 발생.
          Leaky ReLU는 음수에서도 0.1×x로 작은 기울기를 유지 → dying ReLU 방지.

          [Kaiming 초기화]
          Leaky ReLU에 맞게 설계된 초기화 방식.
          fan_out 모드: 역전파 시 기울기 분산을 안정적으로 유지.
        """
        if layer_config[0] == 'c':
            kernel_size, out_channels = layer_config[1:3]
            stride = 1 if len(layer_config) == 3 else layer_config[3]

            # same padding: stride=1이면 공간 크기 유지, stride=2이면 절반으로 축소
            padding = ceil((kernel_size - stride) / 2)

            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1) # <--------------------------------------------
            )
            # Leaky ReLU에 최적화된 Kaiming 초기화
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
        return self.layers(x)


# ==============================================================================
# YOLOv1
# ==============================================================================

class YOLOv1(nn.Module):
    """
    [YOLO의 핵심 설계 철학]
    기존 R-CNN 계열은 "후보 영역 생성 → 각 영역 분류" 2단계로 동작한다.
    YOLO는 이 전체를 단일 합성곱 신경망 하나로 통합한다.
    → 이미지 전체를 단 한 번만 통과(forward pass)시켜 탐지 완료

    [네트워크 구조 개요]
                         ┌──────────────────────────────┐
                         │   Backbone (공유)             │
                         │   20개 합성곱 레이어           │
                         │   ImageNet 사전학습에 사용     │
                         └──────────┬───────────────────┘
                                    │
               ┌────────────────────┴──────────────────┐
               ↓ (classification mode)                  ↓ (detection mode)
    ┌──────────────────────┐              ┌──────────────────────────────┐
    │  분류 헤드            │              │  탐지 헤드                    │
    │  AvgPool → FC(1000)  │              │  4개 conv + LocalConn + FC   │
    │  ImageNet 1000클래스  │              │  출력: 7×7×30 텐서           │
    └──────────────────────┘              └──────────────────────────────┘

    [사전학습 전략]
    1단계: classification 모드로 ImageNet 학습 (입력 224×224)
    2단계: detection 모드로 전환 후 PASCAL VOC 학습 (입력 448×448)
    → 백본의 특징 추출 능력을 최대한 활용하면서 탐지에 특화된 헤드를 추가

    [출력 텐서: 7×7×30]
    각 셀(7×7=49개)이 예측하는 30개 값:
      - 클래스 확률 20개  (C=20, PASCAL VOC 클래스 수)
      - 바운딩 박스 1: [confidence, x, y, w, h]  (5개)
      - 바운딩 박스 2: [confidence, x, y, w, h]  (5개)
    총 20 + 5 + 5 = 30
    """

    # -------------------------------------------------------------------------
    # 백본 설정: 사전학습과 탐지 모두에서 공유되는 20개 합성곱 레이어
    # 논문 Figure 3의 앞부분에 해당
    #
    # [구조 해설]
    # 모듈 1: 7×7 conv(64, stride=2) → MaxPool(2, stride=2)
    #         448×448 → 224×224 → 112×112
    # 모듈 2: 3×3 conv(192) → MaxPool(2, stride=2)
    #         112×112 → 112×112 → 56×56
    # 모듈 3: 1×1(128)→3×3(256)→1×1(256)→3×3(512) → MaxPool
    #         56×56 → 28×28
    # 모듈 4: [1×1(256)→3×3(512)]×4 → 1×1(512)→3×3(1024) → MaxPool
    #         28×28 → 14×14
    # 모듈 5: [1×1(512)→3×3(1024)]×2
    #         14×14 유지 (사전학습은 여기까지 + AvgPool + FC)
    # -------------------------------------------------------------------------
    conv_backbone_config = [
        [('c', 7, 64, 2), ('p', 2, 2)],                                        # 모듈 1
        [('c', 3, 192), ('p', 2, 2)],                                           # 모듈 2
        [('c', 1, 128), ('c', 3, 256), ('c', 1, 256), ('c', 3, 512), ('p', 2, 2)],   # 모듈 3
        [[[('c', 1, 256), ('c', 3, 512)], 4], ('c', 1, 512), ('c', 3, 1024), ('p', 2, 2)],  # 모듈 4
        [[[('c', 1, 512), ('c', 3, 1024)], 2]],                                 # 모듈 5
    ]

    # -------------------------------------------------------------------------
    # 탐지 헤드 설정: 탐지 모드에서만 추가되는 4개 합성곱 레이어
    # 논문 Figure 3의 뒷부분에 해당
    #
    # 모듈 6: 3×3(1024) → 3×3(1024, stride=2)
    #         14×14 → 7×7
    # 모듈 7: 3×3(1024) → 3×3(1024)
    #         7×7 유지
    # -------------------------------------------------------------------------
    conv_detection_config = [
        [('c', 3, 1024), ('c', 3, 1024, 2)],   # 모듈 6: 공간 크기 14→7
        [('c', 3, 1024), ('c', 3, 1024)],       # 모듈 7: 7×7 유지
    ]

    def __init__(self, S: int, B: int, C: int, mode: Optional[str] = 'detection') -> None:
        """
        :param S:    그리드 크기 (S×S로 이미지 분할, 논문에서 S=7)
        :param B:    셀당 예측 바운딩 박스 수 (논문에서 B=2)
        :param C:    클래스 수 (PASCAL VOC: C=20)
        :param mode: 'detection' (파인튜닝) 또는 'classification' (사전학습)
        """
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mode = mode

        # ── 백본 구성 (두 모드 공통) ──────────────────────────────────────────
        backbones_modules_list = []
        in_channels = 3  # RGB 입력
        for module_config in YOLOv1.conv_backbone_config:
            cm = ConvModule(in_channels, module_config)
            backbones_modules_list.append(cm)
            in_channels = cm.out_channels  # 다음 모듈의 입력 채널 수 자동 연결
        self.backbone = nn.Sequential(*backbones_modules_list)
        # 백본 출력: (N, 1024, 14, 14) — detection, (N, 1024, 7, 7) — classification

        # ── 모드별 헤드 구성 ──────────────────────────────────────────────────
        if mode == 'detection':
            # 탐지 합성곱 모듈 (모듈 6, 7)
            head_modules_list = []
            for module_config in YOLOv1.conv_detection_config:
                cm = ConvModule(in_channels, module_config)
                head_modules_list.append(cm)
                in_channels = cm.out_channels
            detection_conv_modules = nn.Sequential(*head_modules_list)
            # 출력: (N, 1024, 7, 7)

            # 탐지 FC 모듈
            # LocallyConnected2d: 7×7 각 위치가 독립적인 가중치로 특징 추출
            # → 일반 FC보다 공간 정보를 더 잘 보존
            # Dropout(0.5): 과적합 방지 (논문에서 명시적으로 언급)
            # 최종 Linear: 1470 = 7×7×(20 + 2×5) = S×S×(C + B×5)
            detection_fc_modules = nn.Sequential(
                LocallyConnected2d(in_channels, 256, 7, 7, 3, 1, 1),  # (N, 256, 7, 7)
                nn.LeakyReLU(0.1),
                nn.Flatten(),                                           # (N, 256×7×7 = 12544)
                nn.Dropout(p=0.5),
                nn.Linear(256 * 7 * 7, S * S * (C + B * 5))           # (N, 1470)
            )

            # LocallyConnected2d 초기화: Kaiming normal
            nn.init.kaiming_normal_(detection_fc_modules[0].weight, a=0.1, mode='fan_out')
            nn.init.zeros_(detection_fc_modules[0].bias)

            self.detection_head = nn.Sequential(detection_conv_modules, detection_fc_modules)
            self.forward = self._forward_detection

        elif mode == 'classification':
            # [사전학습 헤드]
            # AvgPool(7): 7×7 특징맵을 1×1로 압축 → 위치 불변성 확보
            # FC(1024 → C): ImageNet 1000클래스 분류
            # 이 헤드는 파인튜닝 시 제거되고 detection_head로 교체된다
            self.classification_head = nn.Sequential(
                nn.AvgPool2d(7),        # (N, 1024, 7, 7) → (N, 1024, 1, 1)
                nn.Flatten(),           # (N, 1024)
                nn.Linear(1024, C)      # (N, C)
            )
            self.forward = self._forward_classification

        else:
            assert -1

    def _forward_classification(self, x: th.Tensor) -> th.Tensor:
        """
        사전학습 순전파: 백본 → 분류 헤드

        입력:  (N, 3, 224, 224)
        출력:  (N, 1000) — ImageNet 클래스 로짓
        """
        x = self.backbone(x)           # (N, 3, 224, 224) → (N, 1024, 7, 7)
        y = self.classification_head(x) # (N, 1024, 7, 7) → (N, 1000)
        return y

    def _forward_detection(self, x: th.Tensor) -> th.Tensor:
        """
        탐지 순전파: 백본 → 탐지 헤드 → 그리드 텐서 reshape

        [YOLO의 핵심 forward]
        단 한 번의 forward pass로 전체 이미지에서
        모든 물체의 위치와 클래스를 동시에 예측한다.
        R-CNN처럼 후보 영역마다 반복 실행하지 않는다.

        입력:  (N, 3, 448, 448)
        출력:  (N, 7, 7, 30)
               └→ 7×7 그리드의 각 셀이 30개 값을 예측
                  [클래스 확률 20개 | 박스1(5개) | 박스2(5개)]
        """
        x = self.backbone(x)            # (N, 3, 448, 448) → (N, 1024, 14, 14)
        x = self.detection_head(x)      # (N, 1024, 14, 14) → (N, 1470)

        # 1D 벡터를 S×S×(C+B×5) 그리드 형태로 변환
        # reshape: (N, 1470) → (N, 7, 7, 30)
        y = x.reshape(x.shape[0], self.S, self.S, self.C + self.B * 5)
        return y