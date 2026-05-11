import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from typing import Optional, List, Tuple, Union


class LocallyConnected2d(nn.Module):
    """
    Locally Connected 2D Layer.

    일반 Conv2d와 유사하지만 **가중치를 공유하지 않는다.**
    각 윈도우 위치마다 독립적인 가중치를 가진다.

    YOLO v1에서는 논문의 첫 번째 FC layer 대신 이 레이어를 사용한다.
    FC layer와 달리 공간 정보를 일부 보존하면서도
    이미지 전체 맥락(글로벌 컨텍스트)을 참조할 수 있다.

    [논문과의 차이]
    논문 원본: FC layer 사용
    이 구현체: Locally Connected Layer로 대체 (Darknet 공식 구현 기준)
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

        입력 텐서 shape: (N, C, H, W)
        출력 텐서 shape: (N, C', H', W')
            - C' = out_channels
            - H' = floor( (H + 2*padding - kernel_size) / stride + 1 )
            - W' = floor( (W + 2*padding - kernel_size) / stride + 1 )

        :param in_channels:  입력 채널 수
        :param out_channels: 출력 채널 수 (필터 수)
        :param input_h:      입력 텐서의 높이 H
        :param input_w:      입력 텐서의 너비 W
        :param kernel_size:  커널 크기 (각 필터: C × kernel_size × kernel_size)
        :param stride:       슬라이딩 보폭
        :param padding:      상하좌우 패딩 크기
        """
        super(LocallyConnected2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_h = floor((input_h + 2 * padding - kernel_size) / stride + 1)
        self.output_w = floor((input_w + 2 * padding - kernel_size) / stride + 1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # weight shape: (1, in_channels, out_channels, H', W', kernel_size, kernel_size)
        # 위치마다 독립적인 가중치를 가지므로 H'×W' 차원이 추가됨
        self.weight = nn.Parameter(th.randn(1, self.in_channels, self.out_channels,
                                            self.output_h, self.output_w,
                                            self.kernel_size, self.kernel_size))

        # bias shape: (1, out_channels, H', W')
        # 각 (위치, 출력 채널) 조합마다 독립적인 bias
        self.bias = nn.Parameter(th.randn(1, self.out_channels, self.output_h, self.output_w))

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Conv2d처럼 윈도우를 추출하되, 각 위치마다 독립적인 가중치로 연산한다.

        :param x: 입력 텐서
        :return:  출력 텐서
        """
        x = F.pad(x, (self.padding,) * 4)
        # unfold로 슬라이딩 윈도우 추출 후 위치별 독립 가중치와 element-wise 곱
        windows = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)[:, :, None, ...]
        y = th.sum(self.weight * windows, dim=[1, 5, 6]) + self.bias
        return y


class ConvModule(nn.Module):
    """
    YOLO v1 네트워크의 Convolution Block을 구성하는 모듈.

    논문 Network Design 섹션의 각 블록에 해당한다.
    각 Conv layer는 다음 순서로 구성된다:
        Conv2d → BatchNorm2d → LeakyReLU(α=0.1)

    [설계 선택]
    - BatchNorm: 논문 원본에는 없으나 Darknet 구현에서 추가됨
      Conv 이후 BN을 적용하므로 Conv의 bias는 제거 (어차피 BN에서 상쇄됨)
    - LeakyReLU(α=0.1): 음수 영역에서 gradient가 완전히 소멸하는
      일반 ReLU의 dying ReLU 문제를 완화
    - Padding: 'same' padding 방식으로 stride=1일 때 공간 크기 유지
      p = ceil((kernel_size - stride) / 2)
    """

    def __init__(self, in_channels: int, module_config: List[Union[List, Tuple]]) -> None:
        """
        module_config로 레이어 구성을 정의한다.

        레이어 표기 규칙:
            Conv layer  : ('c', kernel_size, out_channels, stride)  # stride 생략 시 1
            MaxPool     : ('p', kernel_size, stride)
            반복 블록   : [[layer_1, ..., layer_m], 반복횟수]

        예시 (Block 4: 1×1→3×3 패턴 4회 반복):
            [[[('c', 1, 256), ('c', 3, 512)], 4], ('c', 1, 512), ('c', 3, 1024), ('p', 2, 2)]

        :param in_channels:   입력 채널 수
        :param module_config: 레이어 구성 리스트
        """
        super(ConvModule, self).__init__()

        self.layers = []
        for sm_config in module_config:
            if isinstance(sm_config, tuple):
                in_channels = self._add_layer(in_channels, sm_config)
            elif isinstance(sm_config, list):
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
        Conv layer 또는 MaxPool layer를 추가한다.

        Conv layer 구성: Conv2d → BatchNorm2d → LeakyReLU(0.1)
        가중치 초기화: Kaiming Normal (LeakyReLU에 맞는 초기화 방식)

        :param in_channels:  현재 레이어의 입력 채널 수
        :param layer_config: 레이어 설정 튜플
        :return: 현재 레이어의 출력 채널 수 (다음 레이어의 입력 채널)
        """
        if layer_config[0] == 'c':
            kernel_size, out_channels = layer_config[1:3]
            stride = 1 if len(layer_config) == 3 else layer_config[3]

            # 공간 크기를 유지하기 위한 'same' padding
            # stride=2이면 공간 크기가 절반으로 줄어듦 (다운샘플링)
            padding = ceil((kernel_size - stride) / 2)

            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                # bias=False: BN이 bias 역할을 대신하므로 불필요
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
                # α=0.1: 음수 입력에도 작은 gradient를 유지 (dying ReLU 방지)
            )
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
        레이어를 순차적으로 통과한다.

        :param x: 입력 텐서
        :return:  출력 텐서
        """
        return self.layers(x)


class YOLOv1(nn.Module):
    """
    YOLO v1 모델 (Redmon et al., CVPR 2015)

    [핵심 아이디어]
    객체 탐지를 단일 신경망의 회귀(Regression) 문제로 재정의한다.
    이미지를 S×S 그리드로 분할하고, 각 셀에서 B개의 bbox와
    C개의 class probability를 동시에 예측한다.

    [출력 텐서]
    shape: (N, S, S, B*5 + C) = (N, 7, 7, 30)  # 기본값 기준
        - B*5: bbox당 [x, y, w, h, confidence]
               confidence = Pr(Object) × IoU  (위치 정확도까지 반영)
        - C  : 각 셀의 class probability (셀당 하나, bbox끼리 공유)

    [2단계 학습]
    1. 사전학습 (classification mode):
       backbone + classification head로 ImageNet 1000-class 분류 학습
       입력: 224×224

    2. Fine-tuning (detection mode):
       backbone + detection head로 PASCAL VOC 탐지 학습
       입력: 448×448 (사전학습보다 해상도를 높여 위치 정보 보강)

    [네트워크 구조 요약]
    Backbone  : 5개 ConvModule (24개 Conv layer 중 앞부분)
    Det. Head : 2개 ConvModule + LocallyConnected2d + FC

    [논문과의 차이점]
    - 첫 FC layer → Locally Connected Layer로 대체 (Darknet 공식 구현 기준)
    - 각 Conv layer에 BatchNorm 추가
    - 학습률 스케줄 및 max_batches 조정
    """

    # -------------------------------------------------------------------------
    # Backbone 설정 (분류/탐지 공통)
    # 논문 Figure 3의 앞부분에 해당
    # 각 블록: 448→224→112→56→28→14 (stride=2로 6번 다운샘플링 → 최종 7×7)
    # -------------------------------------------------------------------------
    conv_backbone_config = [
        # Block 1: 448×448×3 → 224×224×64
        # 7×7 Conv, stride=2로 공간 크기 절반 + MaxPool로 추가 절반
        [('c', 7, 64, 2), ('p', 2, 2)],

        # Block 2: 112×112×64 → 56×56×192
        [('c', 3, 192), ('p', 2, 2)],

        # Block 3: 56×56×192 → 28×28×512
        # 1×1 Conv(채널 축소) → 3×3 Conv(피처 추출) 패턴 사용
        # GoogLeNet Inception과 유사한 철학 (연산량 감소 + 피처 추출)
        [('c', 1, 128), ('c', 3, 256), ('c', 1, 256), ('c', 3, 512), ('p', 2, 2)],

        # Block 4: 28×28×512 → 14×14×1024
        # 1×1(256)→3×3(512) 패턴 4회 반복
        [[[('c', 1, 256), ('c', 3, 512)], 4], ('c', 1, 512), ('c', 3, 1024), ('p', 2, 2)],

        # Block 5: 14×14×1024 → 14×14×1024
        # 1×1(512)→3×3(1024) 패턴 2회 반복
        [[[('c', 1, 512), ('c', 3, 1024)], 2]]
    ]

    # -------------------------------------------------------------------------
    # Detection Head 설정 (탐지 전용)
    # 논문 Figure 3의 뒷부분에 해당
    # -------------------------------------------------------------------------
    conv_detection_config = [
        # Block 6: 14×14×1024 → 7×7×1024
        # stride=2 Conv로 공간 크기 절반 (14→7)
        [('c', 3, 1024), ('c', 3, 1024, 2)],

        # Block 7: 7×7×1024 → 7×7×1024
        [('c', 3, 1024), ('c', 3, 1024)]
    ]

    def __init__(self, S: int, B: int, C: int, mode: Optional[str] = 'detection') -> None:
        """
        YOLO v1 모델 초기화.

        mode에 따라 네트워크 구조가 달라진다:
            'detection'     : backbone + detection head
                              출력 shape: (N, S, S, C + B*5)
            'classification': backbone + classification head
                              출력 shape: (N, C)

        :param S:    그리드 크기 (이미지를 S×S로 분할). 논문 기본값: 7
        :param B:    셀당 예측 bbox 수. 논문 기본값: 2
                     2개를 예측하는 이유: 서로 다른 종횡비의 객체를 커버하기 위함
        :param C:    클래스 수. PASCAL VOC: 20
        :param mode: 'detection' (탐지 fine-tuning) 또는 'classification' (사전학습)
        """
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mode = mode

        # --- Backbone 구성 ---
        # 분류/탐지 모드 공통으로 사용되는 Conv 블록
        # ImageNet 사전학습 후 가중치를 detection fine-tuning에 재사용
        backbones_modules_list = []
        in_channels = 3
        for module_config in YOLOv1.conv_backbone_config:
            cm = ConvModule(in_channels, module_config)
            backbones_modules_list.append(cm)
            in_channels = cm.out_channels
        self.backbone = nn.Sequential(*backbones_modules_list)

        if mode == 'detection':
            # --- Detection Head 구성 ---
            # Conv 블록 (7×7×1024 유지)
            head_modules_list = []
            for module_config in YOLOv1.conv_detection_config:
                cm = ConvModule(in_channels, module_config)
                head_modules_list.append(cm)
                in_channels = cm.out_channels
            detection_conv_modules = nn.Sequential(*head_modules_list)

            # Locally Connected Layer + FC
            # LocallyConnected2d: 논문의 FC layer를 대체
            #   → 위치별 독립 가중치로 공간 정보를 일부 보존하면서 글로벌 추론
            # Flatten 후 FC: 7×7×256 → S×S×(C + B×5)
            #   = 7×7×30 (S=7, B=2, C=20 기준)
            detection_fc_modules = nn.Sequential(
                LocallyConnected2d(in_channels, 256, 7, 7, 3, 1, 1),
                nn.LeakyReLU(0.1),
                nn.Flatten(),
                nn.Dropout(p=0.5),  # 과적합 방지
                nn.Linear(256 * 7 * 7, S * S * (C + B * 5))
                # 출력: S*S*(C + B*5) = 7*7*30 = 1470개의 값
                # 이후 reshape으로 (N, S, S, C+B*5) 텐서로 변환
            )

            nn.init.kaiming_normal_(detection_fc_modules[0].weight, a=0.1, mode='fan_out')
            nn.init.zeros_(detection_fc_modules[0].bias)

            self.detection_head = nn.Sequential(detection_conv_modules,
                                                detection_fc_modules)
            self.forward = self._forward_detection

        elif mode == 'classification':
            # --- Classification Head 구성 ---
            # 사전학습용: backbone 출력(7×7×1024)을 GlobalAvgPool → FC로 분류
            self.classification_head = nn.Sequential(
                nn.AvgPool2d(7),   # 7×7×1024 → 1×1×1024 (Global Average Pooling)
                nn.Flatten(),      # 1024
                nn.Linear(1024, C) # 1024 → C (ImageNet: 1000)
            )
            self.forward = self._forward_classification

        else:
            assert -1

    def _forward_classification(self, x: th.Tensor) -> th.Tensor:
        """
        사전학습(분류) 모드 Forward Pass.

        backbone → classification head 순서로 통과.
        ImageNet 1000-class 분류 학습에 사용.

        :param x: 입력 이미지 텐서 (N, 3, 224, 224)
        :return:  클래스 logit (N, C)
        """
        x = self.backbone(x)
        y = self.classification_head(x)
        return y

    def _forward_detection(self, x: th.Tensor) -> th.Tensor:
        """
        탐지(Detection) 모드 Forward Pass.

        backbone → detection head → reshape 순서로 통과.

        최종 출력 텐서 shape: (N, S, S, C + B*5)
            예: (N, 7, 7, 30) when S=7, B=2, C=20

        각 셀(i, j)의 출력 벡터 구성 (30차원):
            [x1, y1, w1, h1, conf1,  # bbox 1
             x2, y2, w2, h2, conf2,  # bbox 2
             p1, p2, ..., p20]       # class probabilities (셀 전체 공유)

            x, y      : 셀 기준 bbox 중심 좌표 (0~1 정규화)
            w, h      : 이미지 전체 기준 bbox 크기 (0~1 정규화)
            confidence: Pr(Object) × IoU  ← 위치 정확도까지 반영
            p1~p20    : 각 클래스일 확률 (셀당 하나, bbox끼리 공유)

        [한계]
        class probability가 셀당 하나이므로,
        하나의 셀에 서로 다른 클래스의 객체가 있으면 하나만 탐지 가능.
        → YOLO v2에서 anchor box 도입으로 해결

        :param x: 입력 이미지 텐서 (N, 3, 448, 448)
        :return:  탐지 결과 텐서 (N, S, S, C + B*5)
        """
        x = self.backbone(x)
        x = self.detection_head(x)
        # 1D 출력을 그리드 구조로 reshape
        # (N, S*S*(C+B*5)) → (N, S, S, C+B*5)
        y = x.reshape(x.shape[0], self.S, self.S, self.C + self.B * 5)
        return y
