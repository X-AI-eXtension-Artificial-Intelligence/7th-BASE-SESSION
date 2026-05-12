import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from typing import Optional, List, Tuple, Union


class LocallyConnected2d(nn.Module):
    """
    LocallyConnected2d는 Conv2d와 비슷하게 작동하지만, 중요한 차이 있음
    일반 convolution은 같은 filter를 모든 위치에 공유해서 사용하지만, LocallyConnected2d는 위치마다 다른 weight를 사용
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
        필요한 파라미터 값들을 설명함
        입력 tensor = (N, C, H, W)
        - N → batch size
        - C → input channel 수
        - H → input height
        - W → input width

        출력 tensor = (N, C', H', W')
        - C' → output channel 수
        - H' → output feature map 높이
        - W' → output feature map 너비
        출력 크기 convolution 공식
        -> H' = floor((H + 2 * padding - kernel_size) / stride + 1)
        -> W' = floor((W + 2 * padding - kernel_size) / stride + 1)
        """
        super(LocallyConnected2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_h = floor((input_h + 2 * padding - kernel_size) / stride + 1)
        self.output_w = floor((input_w + 2 * padding - kernel_size) / stride + 1)
        self.kernel_size = kernel_size # filter 크기
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(th.randn(1, self.in_channels, self.out_channels,
                                            self.output_h, self.output_w,
                                            self.kernel_size, self.kernel_size))

        self.bias = nn.Parameter(th.randn(1, self.out_channels, self.output_h, self.output_w))

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        입력 이미지에서 window, 즉 작은 patch를 뽑는 방식은 convolution과 같습니다.

        하지만 convolution과 달리 각 window 위치마다 서로 다른 weight를 곱합니다.

        또한 각 위치와 output channel 조합마다 다른 bias도 더합니다.
        """
        x = F.pad(x, (self.padding,) * 4)
        windows = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)[:, :, None, ...]
        y = th.sum(self.weight * windows, dim=[1, 5, 6]) + self.bias
        return y


class ConvModule(nn.Module):
    """
    여러 개의 convolution layer와 max pooling layer를 설정값에 따라 자동으로 쌓아주는 역할
    """

    def __init__(self, in_channels: int, module_config: List[Union[List, Tuple]]) -> None:
        """
        convolution layer = ('c', kernel_size, out_channels, stride)
        -> stride는 기본값 1
        pooling layer = ('p', kernel_size, stride)
        반복 구조 = [[layer_1, ..., layer_m], k]
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
        convolution layer의 구성 요소
        same 2D convolution
        → Batch Normalization
        → Leaky ReLU

        BatchNorm이 뒤에 오면 bias 효과가 BatchNorm에 의해 상쇄될 수 있으므로 Conv2d의 bias는 사용하지 않음
        MaxPool layer는 단순히 MaxPool 연산 하나만 포함
        """
        if layer_config[0] == 'c':
            kernel_size, out_channels = layer_config[1:3]
            stride = 1 if len(layer_config) == 3 else layer_config[3]
            padding = ceil((kernel_size - stride) / 2)

            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                            bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.LeakyReLU(0.1))
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
        입력값 x는 ConvModule 안의 layer 들을 순서대로 통과함
        """
        return self.layers(x)


class YOLOv1(nn.Module):
    """
    이 모델은 먼저 ImageNet classification task로 pretraining하고,
    이후 PASCAL VOC object detection task로 fine-tuning한다.

    conv_backbone_config
    → classification과 detection에 공통으로 쓰이는 backbone 구조

    conv_detection_config
    → detection task에서만 쓰이는 추가 convolution 구조
    """
    conv_backbone_config = [[('c', 7, 64, 2), ('p', 2, 2)],
                            [('c', 3, 192), ('p', 2, 2)],
                            [('c', 1, 128), ('c', 3, 256), ('c', 1, 256), ('c', 3, 512), ('p', 2, 2)],
                            [[[('c', 1, 256), ('c', 3, 512)], 4], ('c', 1, 512), ('c', 3, 1024), ('p', 2, 2)],
                            [[[('c', 1, 512), ('c', 3, 1024)], 2]]]

    conv_detection_config = [[('c', 3, 1024), ('c', 3, 1024, 2)],
                             [('c', 3, 1024), ('c', 3, 1024)]]

    def __init__(self, S: int, B: int, C: int, mode: Optional[str] = 'detection') -> None:
        """
        YOLOv1 모델을 만들 때 mode에 따라 구조가 달라짐
        위에 언급한
        detection       → object detection fine-tuning용
        classification  → classification pretraining용
        """
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mode = mode

        backbones_modules_list = []
        in_channels = 3
        for module_config in YOLOv1.conv_backbone_config:
            cm = ConvModule(in_channels, module_config)
            backbones_modules_list.append(cm)
            in_channels = cm.out_channels
        self.backbone = nn.Sequential(*backbones_modules_list)

        if mode == 'detection':
            head_modules_list = []
            for module_config in YOLOv1.conv_detection_config:
                cm = ConvModule(in_channels, module_config)
                head_modules_list.append(cm)
                in_channels = cm.out_channels
            detection_conv_modules = nn.Sequential(*head_modules_list)
            detection_fc_modules = nn.Sequential(LocallyConnected2d(in_channels, 256, 7, 7, 3, 1, 1),
                                                 nn.LeakyReLU(0.1),
                                                 nn.Flatten(),
                                                 nn.Dropout(p=0.5),
                                                 nn.Linear(256 * 7 * 7, S * S * (C + B * 5)))

            nn.init.kaiming_normal_(detection_fc_modules[0].weight, a=0.1, mode='fan_out')
            nn.init.zeros_(detection_fc_modules[0].bias)

            self.detection_head = nn.Sequential(detection_conv_modules,
                                                detection_fc_modules)
            self.forward = self._forward_detection

        elif mode == 'classification':
            self.classification_head = nn.Sequential(nn.AvgPool2d(7),
                                                     nn.Flatten(),
                                                     nn.Linear(1024, C))
            self.forward = self._forward_classification

        else:
            assert -1

    def _forward_classification(self, x: th.Tensor) -> th.Tensor:
        """
        classification mode에서의 forward
        
        입력 이미지
        → backbone
        → classification head
        → class score 출력
        = detection box가 아니라 이미지 전체의 class 예측값을 반환
        """
        x = self.backbone(x)
        y = self.classification_head(x)
        return y

    def _forward_detection(self, x: th.Tensor) -> th.Tensor:
        """
        detection mode에서의 forward

        입력 이미지
        → backbone
        → detection head
        → YOLO detection output
        = 반환값은 object detection용 출력
        """
        x = self.backbone(x)
        x = self.detection_head(x)
        y = x.reshape(x.shape[0], self.S, self.S, self.C + self.B * 5)
        return y
