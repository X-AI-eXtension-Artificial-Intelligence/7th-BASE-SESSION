import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from typing import Optional, List, Tuple, Union


class LocallyConnected2d(nn.Module):
    """
    Conv2d와 비슷하지만 weight를 공유하지 않는 layer

    일반 Conv:
        같은 filter를 모든 위치에서 공유

    LocallyConnected:
        위치마다 다른 filter 사용
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

        # 입력 channel 수
        self.in_channels = in_channels

        # 출력 channel 수
        self.out_channels = out_channels

        # 출력 feature map 높이 계산
        self.output_h = floor(
            (input_h + 2 * padding - kernel_size) / stride + 1
        )

        # 출력 feature map 너비 계산
        self.output_w = floor(
            (input_w + 2 * padding - kernel_size) / stride + 1
        )

        # kernel 크기
        self.kernel_size = kernel_size

        # stride
        self.stride = stride

        # padding
        self.padding = padding

        """
        위치마다 다른 weight 사용

        shape:
            [
                1,
                in_channels,
                out_channels,
                output_h,
                output_w,
                kernel_h,
                kernel_w
            ]
        """

        self.weight = nn.Parameter(
            th.randn(
                1,
                self.in_channels,
                self.out_channels,
                self.output_h,
                self.output_w,
                self.kernel_size,
                self.kernel_size
            )
        )

        # 위치마다 bias도 따로 존재
        self.bias = nn.Parameter(
            th.randn(
                1,
                self.out_channels,
                self.output_h,
                self.output_w
            )
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        x shape:
            [batch, channels, H, W]
        """

        # padding 적용
        x = F.pad(x, (self.padding,) * 4)

        """
        sliding window 추출

        unfold(2):
            height 방향 sliding

        unfold(3):
            width 방향 sliding
        """

        windows = x.unfold(
            2,
            self.kernel_size,
            self.stride
        ).unfold(
            3,
            self.kernel_size,
            self.stride
        )

        # broadcasting 위한 차원 추가
        windows = windows[:, :, None, ...]

        # 각 위치별 weight 적용
        y = th.sum(
            self.weight * windows,
            dim=[1, 5, 6]
        ) + self.bias

        return y


class ConvModule(nn.Module):
    """
    YOLO 논문의 convolution module 구현

    Conv -> BN -> LeakyReLU 구조 사용
    """

    def __init__(self,
                 in_channels: int,
                 module_config: List[Union[List, Tuple]]) -> None:

        super(ConvModule, self).__init__()

        # layer 저장용 list
        self.layers = []

        """
        module_config 예시:

        ('c', 3, 64)
            -> Conv layer

        ('p', 2, 2)
            -> MaxPool

        [[(...), (...)], 4]
            -> block 반복
        """

        for sm_config in module_config:

            # 단일 layer
            if isinstance(sm_config, tuple):

                in_channels = self._add_layer(
                    in_channels,
                    sm_config
                )

            # 반복 구조
            elif isinstance(sm_config, list):

                sm_layers, r = sm_config

                # r번 반복
                for _ in range(r):

                    for layer_config in sm_layers:

                        in_channels = self._add_layer(
                            in_channels,
                            layer_config
                        )

            else:
                assert -1

        # 마지막 output channel 저장
        self.out_channels = in_channels

        # Sequential 변환
        self.layers = nn.Sequential(*self.layers)

    def _add_layer(self,
                   in_channels: int,
                   layer_config: Tuple) -> int:
        """
        layer 추가 함수
        """

        # ------------------------
        # Conv Layer
        # ------------------------

        if layer_config[0] == 'c':

            # kernel size, output channel
            kernel_size, out_channels = layer_config[1:3]

            # stride 없으면 기본 1
            stride = 1 if len(layer_config) == 3 else layer_config[3]

            """
            same convolution 비슷하게 padding 계산
            """

            padding = ceil((kernel_size - stride) / 2)

            # Conv + BN + LeakyReLU
            layer = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False
                ),

                nn.BatchNorm2d(out_channels),

                nn.LeakyReLU(0.1)
            )

            # He initialization
            nn.init.kaiming_normal_(
                layer[0].weight,
                a=0.1,
                mode='fan_out',
                nonlinearity='leaky_relu'
            )

            # layer 저장
            self.layers.append(layer)

            # 다음 입력 channel 업데이트
            in_channels = out_channels

        # ------------------------
        # MaxPool Layer
        # ------------------------

        elif layer_config[0] == 'p':

            kernel_size, stride = layer_config[1:]

            self.layers.append(
                nn.MaxPool2d(
                    kernel_size,
                    stride
                )
            )

        else:
            assert -1

        return in_channels

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        layer 순차 실행
        """

        return self.layers(x)


class YOLOv1(nn.Module):
    """
    YOLOv1 모델

    mode='classification'
        -> ImageNet pretraining

    mode='detection'
        -> VOC detection fine-tuning
    """

    # backbone 구조
    conv_backbone_config = [

        [('c', 7, 64, 2), ('p', 2, 2)],

        [('c', 3, 192), ('p', 2, 2)],

        [
            ('c', 1, 128),
            ('c', 3, 256),
            ('c', 1, 256),
            ('c', 3, 512),
            ('p', 2, 2)
        ],

        [
            [[('c', 1, 256), ('c', 3, 512)], 4],
            ('c', 1, 512),
            ('c', 3, 1024),
            ('p', 2, 2)
        ],

        [
            [[('c', 1, 512), ('c', 3, 1024)], 2]
        ]
    ]

    # detection head 구조
    conv_detection_config = [

        [('c', 3, 1024), ('c', 3, 1024, 2)],

        [('c', 3, 1024), ('c', 3, 1024)]
    ]

    def __init__(self,
                 S: int,
                 B: int,
                 C: int,
                 mode: Optional[str] = 'detection') -> None:

        super(YOLOv1, self).__init__()

        # grid 개수
        self.S = S

        # bbox 개수
        self.B = B

        # class 개수
        self.C = C

        self.mode = mode

        # =================================
        # Backbone 생성
        # =================================

        backbones_modules_list = []

        # RGB image 입력
        in_channels = 3

        for module_config in YOLOv1.conv_backbone_config:

            cm = ConvModule(
                in_channels,
                module_config
            )

            backbones_modules_list.append(cm)

            in_channels = cm.out_channels

        # backbone 생성
        self.backbone = nn.Sequential(
            *backbones_modules_list
        )

        # =================================
        # Detection Mode
        # =================================

        if mode == 'detection':

            head_modules_list = []

            for module_config in YOLOv1.conv_detection_config:

                cm = ConvModule(
                    in_channels,
                    module_config
                )

                head_modules_list.append(cm)

                in_channels = cm.out_channels

            # detection conv head
            detection_conv_modules = nn.Sequential(
                *head_modules_list
            )

            """
            Fully Connected Detection Head

            최종 출력:
                S*S*(C+B*5)
            """

            detection_fc_modules = nn.Sequential(

                # locally connected layer
                LocallyConnected2d(
                    in_channels,
                    256,
                    7,
                    7,
                    3,
                    1,
                    1
                ),

                nn.LeakyReLU(0.1),

                # flatten
                nn.Flatten(),

                # dropout
                nn.Dropout(p=0.5),

                # 최종 detection output
                nn.Linear(
                    256 * 7 * 7,
                    S * S * (C + B * 5)
                )
            )

            # weight initialization
            nn.init.kaiming_normal_(
                detection_fc_modules[0].weight,
                a=0.1,
                mode='fan_out'
            )

            # bias 초기화
            nn.init.zeros_(
                detection_fc_modules[0].bias
            )

            # detection head 생성
            self.detection_head = nn.Sequential(
                detection_conv_modules,
                detection_fc_modules
            )

            # detection forward 사용
            self.forward = self._forward_detection

        # =================================
        # Classification Mode
        # =================================

        elif mode == 'classification':

            self.classification_head = nn.Sequential(

                # global average pooling
                nn.AvgPool2d(7),

                nn.Flatten(),

                # 최종 classification
                nn.Linear(1024, C)
            )

            # classification forward 사용
            self.forward = self._forward_classification

        else:
            assert -1

    def _forward_classification(
            self,
            x: th.Tensor
    ) -> th.Tensor:
        """
        classification forward
        """

        # backbone
        x = self.backbone(x)

        # classification head
        y = self.classification_head(x)

        return y

    def _forward_detection(
            self,
            x: th.Tensor
    ) -> th.Tensor:
        """
        detection forward
        """

        # backbone
        x = self.backbone(x)

        # detection head
        x = self.detection_head(x)

        """
        reshape

        [batch, 1470]
            ->
        [batch, 7, 7, 30]
        """

        y = x.reshape(
            x.shape[0],
            self.S,
            self.S,
            self.C + self.B * 5
        )

        return y