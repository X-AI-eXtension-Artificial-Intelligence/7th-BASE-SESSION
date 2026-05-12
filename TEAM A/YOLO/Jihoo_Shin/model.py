import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from typing import Optional, List, Tuple, Union


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, input_h, input_w,
                 kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_h = floor((input_h + 2 * padding - kernel_size) / stride + 1)
        self.output_w = floor((input_w + 2 * padding - kernel_size) / stride + 1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(th.randn(1, self.in_channels, self.out_channels,
                                            self.output_h, self.output_w,
                                            self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(th.randn(1, self.out_channels, self.output_h, self.output_w))

    def forward(self, x):
        x = F.pad(x, (self.padding,) * 4)
        windows = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)[:, :, None, ...]
        y = th.sum(self.weight * windows, dim=[1, 5, 6]) + self.bias
        return y


class ConvModule(nn.Module):
    def __init__(self, in_channels, module_config):
        super().__init__()
        self.layers = []
        for sm_config in module_config:
            if isinstance(sm_config, tuple):
                in_channels = self._add_layer(in_channels, sm_config)
            elif isinstance(sm_config, list):
                sm_layers, r = sm_config
                for _ in range(r):
                    for layer_config in sm_layers:
                        in_channels = self._add_layer(in_channels, layer_config)
        self.out_channels = in_channels
        self.layers = nn.Sequential(*self.layers)

    def _add_layer(self, in_channels, layer_config):
        if layer_config[0] == 'c':
            kernel_size, out_channels = layer_config[1:3]
            stride = 1 if len(layer_config) == 3 else layer_config[3]
            padding = ceil((kernel_size - stride) / 2)
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            )
            nn.init.kaiming_normal_(layer[0].weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            self.layers.append(layer)
            in_channels = out_channels
        elif layer_config[0] == 'p':
            kernel_size, stride = layer_config[1:]
            self.layers.append(nn.MaxPool2d(kernel_size, stride))
        return in_channels

    def forward(self, x):
        return self.layers(x)


class YOLOv1(nn.Module):
    conv_backbone_config = [
        [('c', 7, 64, 2), ('p', 2, 2)],
        [('c', 3, 192), ('p', 2, 2)],
        [('c', 1, 128), ('c', 3, 256), ('c', 1, 256), ('c', 3, 512), ('p', 2, 2)],
        [[[('c', 1, 256), ('c', 3, 512)], 4], ('c', 1, 512), ('c', 3, 1024), ('p', 2, 2)],
        [[[('c', 1, 512), ('c', 3, 1024)], 2]]
    ]
    conv_detection_config = [
        [('c', 3, 1024), ('c', 3, 1024, 2)],
        [('c', 3, 1024), ('c', 3, 1024)]
    ]

    def __init__(self, S=7, B=2, C=20, mode='detection'):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mode = mode

        in_channels = 3
        backbones = []
        for cfg in self.conv_backbone_config:
            cm = ConvModule(in_channels, cfg)
            backbones.append(cm)
            in_channels = cm.out_channels
        self.backbone = nn.Sequential(*backbones)

        if mode == 'detection':
            head_modules = []
            for cfg in self.conv_detection_config:
                cm = ConvModule(in_channels, cfg)
                head_modules.append(cm)
                in_channels = cm.out_channels
            detection_conv = nn.Sequential(*head_modules)
            detection_fc = nn.Sequential(
                LocallyConnected2d(in_channels, 256, 7, 7, 3, 1, 1),
                nn.LeakyReLU(0.1),
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(256 * 7 * 7, S * S * (C + B * 5))
            )
            nn.init.kaiming_normal_(detection_fc[0].weight, a=0.1, mode='fan_out')
            nn.init.zeros_(detection_fc[0].bias)
            self.detection_head = nn.Sequential(detection_conv, detection_fc)
            self.forward = self._forward_detection

        elif mode == 'classification':
            self.classification_head = nn.Sequential(
                nn.AvgPool2d(7), nn.Flatten(), nn.Linear(1024, C)
            )
            self.forward = self._forward_classification

    def _forward_classification(self, x):
        return self.classification_head(self.backbone(x))

    def _forward_detection(self, x):
        x = self.backbone(x)
        x = self.detection_head(x)
        return x.reshape(x.shape[0], self.S, self.S, self.C + self.B * 5)
