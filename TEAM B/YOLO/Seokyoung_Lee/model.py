import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from typing import Optional, List, Tuple, Union


# ============================================================
# [model.py 개요]
# 논문 Figure 3의 YOLOv1 네트워크 구조를 구현한 파일입니다.
# 세 개의 클래스로 구성됩니다:
#
#   1. LocallyConnected2d : detection head의 첫 번째 FC 레이어
#   2. ConvModule         : config 기반으로 conv/pool 레이어를 조립하는 빌딩 블록
#   3. YOLOv1             : backbone + head를 조합한 전체 모델
#
# 모델은 두 가지 mode로 동작합니다:
#   - 'classification' : pretrain.py에서 ImageNet pretraining에 사용
#   - 'detection'      : train.py에서 VOC detection fine-tuning에 사용
# ============================================================


# ============================================================
# [LocallyConnected2d]
# 논문 Figure 3의 detection head에서 FC 레이어 직전에 위치하는
# "Conn. Layer"를 구현합니다.
#
# Conv2d와 동작은 유사하지만, 가중치를 공유하지 않습니다.
# 각 위치(H', W')마다 독립적인 가중치 셋을 가집니다.
# 이를 통해 grid cell별로 다른 특징을 학습할 수 있습니다.
# ============================================================
class LocallyConnected2d(nn.Module):
    """
    A Locally Connected 2D Layer behaves like a 2D Convolution Module, with the important distinction that the weights
    are not shared. Instead, each window position has its own set of weights.
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
        Initialize the Locally Connected 2D Layer. The input of the layer is a Tensor with dimensions (N, C, H, W)
        and the output is a Tensor with dimensions (N, C', H', W'), where:

            - C' = output_channels
            - H' = floor( (H + 2 * padding - kernel_size) / stride + 1 )
            - W' = floor( (W + 2 * padding - kernel_size) / stride + 1 )

        :param in_channels: The input channels of the Locally Connected 2D layer
        :param out_channels: The output channels (#filters) of the Locally Connected 2D layer
        :param input_h: The height H of the input tensors that have shape (N, C, H, W)
        :param input_w: The width W of the input tensors that have shape (N, C, H, W)
        :param kernel_size: The size of the kernel. Each filter has dimensions (C x kernel_size x kernel_size)
        :param stride: The stride based on which the patches are extracted.
        :param padding: The padding that will be applied to the left, right, top and bottom of the input Tensors
        """
        super(LocallyConnected2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_h = floor((input_h + 2 * padding - kernel_size) / stride + 1)
        self.output_w = floor((input_w + 2 * padding - kernel_size) / stride + 1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Conv2d의 weight shape: (out_ch, in_ch, kH, kW)
        # LocallyConnected2d의 weight shape: (1, in_ch, out_ch, H', W', kH, kW)
        # 위치(H', W')마다 독립적인 (out_ch, in_ch, kH, kW) 가중치를 가집니다.
        self.weight = nn.Parameter(th.randn(1, self.in_channels, self.out_channels,
                                            self.output_h, self.output_w,
                                            self.kernel_size, self.kernel_size))

        # bias shape: (1, out_ch, H', W') — 위치마다 독립적인 bias
        self.bias = nn.Parameter(th.randn(1, self.out_channels, self.output_h, self.output_w))

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        The windows are extracted like in a 2D Convolution layer, but each window is multiplied with its own set of
        weights. A different bias term is also added for each (patch location, output channel)-combination,

        :param x: The input of the Locally Connected 2D Layer
        :return: The output of the Locally Connected 2D Layer
        """
        x = F.pad(x, (self.padding,) * 4)
        # unfold로 sliding window를 추출합니다. (Conv2d의 im2col에 해당)
        # windows shape: (N, in_ch, 1, H', W', kH, kW)
        windows = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)[:, :, None, ...]
        # weight와 element-wise 곱 후 in_ch, kH, kW 차원을 합산합니다.
        # 결과 shape: (N, out_ch, H', W')
        y = th.sum(self.weight * windows, dim=[1, 5, 6]) + self.bias
        return y


# ============================================================
# [ConvModule]
# 논문 Figure 3의 각 conv 블록을 config 리스트로 선언적으로 조립합니다.
#
# config 표기법:
#   ('c', kernel_size, out_channels)          → stride=1 conv
#   ('c', kernel_size, out_channels, stride)  → strided conv
#   ('p', kernel_size, stride)                → MaxPool
#   [[layer1, layer2, ...], k]                → layer1, layer2를 k번 반복
#
# 각 Conv 레이어는 Conv2d + BatchNorm + LeakyReLU(0.1)로 구성됩니다.
# 논문 원본은 BN을 사용하지 않지만, 학습 안정성을 위해 추가되었습니다.
# ============================================================
class ConvModule(nn.Module):
    """
    The ConvModule class implements the convolution modules of the Network Design figure in the paper.
    """

    def __init__(self, in_channels: int, module_config: List[Union[List, Tuple]]) -> None:
        """
        The module configuration argument is a list which describes the layers of the module. Each layer is represented
        as a tuple:

        - The Conv. Layers are represented as ('c', kernel_size, out_channels, stride)
        - The MaxPool Layers are represented as ('p', kernel_size, stride)

        When some layers layer_1, ..., layer_m are repeated for k times within a module, they are represented as
        [[layer_1, ..., layer_m], k]

        The height and the width dimensions n of the representation are reduced only due to the strides of either the
        Conv. Layers or the MaxPool Layers. To this end, a padding p is used in the convolutional layers, with:

        p = ceil((f-s) / 2)

        :param in_channels: The input channels of the convolution module
        :param module_config: The module configuration
        """
        super(ConvModule, self).__init__()

        self.layers = []
        for sm_config in module_config:
            if isinstance(sm_config, tuple):
                # 단일 레이어 config
                in_channels = self._add_layer(in_channels, sm_config)
            elif isinstance(sm_config, list):
                # [[layer1, layer2, ...], k] 형태의 반복 블록
                sm_layers, r = sm_config
                for _ in range(r):
                    for layer_config in sm_layers:
                        in_channels = self._add_layer(in_channels, layer_config)
            else:
                assert -1
        # 마지막 레이어의 출력 채널을 기록해 다음 모듈의 in_channels로 전달합니다.
        self.out_channels = in_channels
        self.layers = nn.Sequential(*self.layers)

    def _add_layer(self, in_channels: int, layer_config: Tuple) -> int:
        """
        Add a Conv. Layer or a MaxPool layer to the layers of the Convolution Module.
        The Convolution layers consist of:
            - a 'same' 2D convolution (with/without stride)
            - a Batch Normalization layer
            - a Leaky ReLU activation function with alpha=0.1

        We do not add the bias term in the convolution operation, as it would be zeroed out by the Batch Normalization
        layer.

        The MaxPool layer contains only a single MaxPool operation.

        :param in_channels: The input channels of the layer
        :param layer_config: A tuple that represents the layer configuration
        :return: The output channels of the layer
        """
        if layer_config[0] == 'c':
            kernel_size, out_channels = layer_config[1:3]
            stride = 1 if len(layer_config) == 3 else layer_config[3]
            # padding = ceil((f-s)/2): stride가 있어도 spatial 해상도가
            # 의도한 만큼만 줄어들도록 'same-like' padding을 계산합니다.
            padding = ceil((kernel_size - stride) / 2)

            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                            bias=False),   # BN이 bias를 흡수하므로 bias=False
                                  nn.BatchNorm2d(out_channels),
                                  nn.LeakyReLU(0.1))
            # Kaiming initialization: LeakyReLU에 적합한 초기화 방식입니다.
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
        The input is propagated through the layers of the convolution module sequentially.

        :param x: The input of the convolution module
        :return: The output of the convolution module
        """
        return self.layers(x)


# ============================================================
# [YOLOv1]
# 논문 Figure 3의 전체 네트워크를 구현합니다.
#
# [backbone]
# conv_backbone_config에 정의된 5개 ConvModule로 구성됩니다.
# pretrain과 detection 모두에서 공유되는 feature extractor입니다.
# 입력 (N, 3, 448, 448) → 출력 (N, 1024, 7, 7)
#
# [classification head] (pretrain 전용)
# AvgPool(7) → Flatten → Linear(1024, C=1000)
# ImageNet 1000-class 분류를 위한 헤드입니다.
#
# [detection head] (detection 전용)
# conv_detection_config의 2개 ConvModule
# → LocallyConnected2d → LeakyReLU → Flatten → Dropout → Linear
# → reshape to (N, S, S, C + B*5)
# 최종 출력이 논문의 7×7×30 prediction tensor에 해당합니다.
# ============================================================
class YOLOv1(nn.Module):
    """
    The YOLOv1 model. The model is pretrained on the Imagenet 2012 Challenge classification task, and it then fine-tuned
    for objection-detection task on the PASCAL VOC dataset.

    The conv_backbone_config consists of the convolution modules that are used both for the classification and the
    detection task.

    The conv_detection_config consists of the convolution modules that are used only for the detection task.

    The last module of the backbone and the first module of the detection head are actually a single module in the
    paper's figure.
    """
    # 논문 Figure 3의 backbone (앞 20개 conv layer에 해당)을 config로 선언합니다.
    # 각 리스트 하나가 하나의 ConvModule(논문의 conv block)에 대응합니다.
    conv_backbone_config = [[('c', 7, 64, 2), ('p', 2, 2)],
                            [('c', 3, 192), ('p', 2, 2)],
                            [('c', 1, 128), ('c', 3, 256), ('c', 1, 256), ('c', 3, 512), ('p', 2, 2)],
                            [[[('c', 1, 256), ('c', 3, 512)], 4], ('c', 1, 512), ('c', 3, 1024), ('p', 2, 2)],
                            [[[('c', 1, 512), ('c', 3, 1024)], 2]]]

    # backbone 이후 detection에만 추가되는 4개 conv layer를 정의합니다.
    conv_detection_config = [[('c', 3, 1024), ('c', 3, 1024, 2)],
                             [('c', 3, 1024), ('c', 3, 1024)]]

    def __init__(self, S: int, B: int, C: int, mode: Optional[str] = 'detection') -> None:
        """
        The YOLO model's initialization. Depending on the mode of the network, the model's architecture is altered.
        Specifically, for the 'detection' mode, the network consists of the backbone and the detection head.
        For the 'classification' mode, the network consists of the backbone and the classification head.

        :param S: The S parameter of the YOLO model. Each image is split into an (S x S) grid.
        :param B: The B parameter of the YOLO model. The model predicts B bounding boxes for each of the S^2 grid cells.
        :param C: The number of classes for the detection or the classification task
        :param mode: The mode of the YOLO model, either 'detection' (fine-tuning) or 'classification'(pre-training)
        """
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mode = mode

        # backbone은 mode에 무관하게 항상 동일하게 구성됩니다.
        backbones_modules_list = []
        in_channels = 3
        for module_config in YOLOv1.conv_backbone_config:
            cm = ConvModule(in_channels, module_config)
            backbones_modules_list.append(cm)
            in_channels = cm.out_channels
        self.backbone = nn.Sequential(*backbones_modules_list)

        if mode == 'detection':
            # detection head: conv 모듈 + FC 모듈
            head_modules_list = []
            for module_config in YOLOv1.conv_detection_config:
                cm = ConvModule(in_channels, module_config)
                head_modules_list.append(cm)
                in_channels = cm.out_channels
            detection_conv_modules = nn.Sequential(*head_modules_list)

            # LocallyConnected2d: (N, 1024, 7, 7) → (N, 256, 7, 7)
            # Dropout(0.5): 논문 Section 2.2에서 언급된 overfitting 방지 기법
            # Linear: (N, 256*7*7) → (N, S*S*(C+B*5))
            detection_fc_modules = nn.Sequential(LocallyConnected2d(in_channels, 256, 7, 7, 3, 1, 1),
                                                 nn.LeakyReLU(0.1),
                                                 nn.Flatten(),
                                                 nn.Dropout(p=0.5),
                                                 nn.Linear(256 * 7 * 7, S * S * (C + B * 5)))

            nn.init.kaiming_normal_(detection_fc_modules[0].weight, a=0.1, mode='fan_out')
            nn.init.zeros_(detection_fc_modules[0].bias)

            self.detection_head = nn.Sequential(detection_conv_modules,
                                                detection_fc_modules)
            # mode에 따라 forward 메서드를 동적으로 할당합니다.
            self.forward = self._forward_detection

        elif mode == 'classification':
            # AvgPool(7): (N, 1024, 7, 7) → (N, 1024, 1, 1)
            # Linear(1024, C): ImageNet 1000-class 분류 헤드
            self.classification_head = nn.Sequential(nn.AvgPool2d(7),
                                                     nn.Flatten(),
                                                     nn.Linear(1024, C))
            self.forward = self._forward_classification

        else:
            assert -1

    def _forward_classification(self, x: th.Tensor) -> th.Tensor:
        """
        The input is propagated through the modules of the backbone and the classification head sequentially.

        :param x: The input of the YOLO model
        :return: The output of the YOLO model for the classification task
        """
        x = self.backbone(x)
        y = self.classification_head(x)
        return y

    def _forward_detection(self, x: th.Tensor) -> th.Tensor:
        """
        The input is propagated through the modules of the backbone and the detection head sequentially.

        :param x: The input of the YOLO model
        :return: The output of the YOLO model for the object detection task
        """
        x = self.backbone(x)
        x = self.detection_head(x)
        # 1D로 펼쳐진 출력을 (N, S, S, C+B*5) grid 형태로 복원합니다.
        # 이 구조가 논문의 S×S×(B*5+C) prediction tensor에 해당합니다.
        y = x.reshape(x.shape[0], self.S, self.S, self.C + self.B * 5)
        return y