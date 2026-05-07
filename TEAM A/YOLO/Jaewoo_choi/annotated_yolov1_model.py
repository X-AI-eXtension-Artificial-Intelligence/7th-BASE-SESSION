# annotated_yolov1_model.py
#
# nsoul97/yolov1_pytorch repository와 YOLO v1 논문 구조를 참고하여 정리한 설명용 코드다.
#
# 핵심 참고:
# - Redmon et al., 2015, You Only Look Once: Unified, Real-Time Object Detection
# - nsoul97/yolov1_pytorch
#
# 이 파일은 과제 제출용 설명 코드다.
# 전체 학습 파이프라인은 Colab end-to-end cell에서 수행했다.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    YOLO 계열 CNN에서 반복적으로 사용되는 Conv-BN-LeakyReLU block이다.

    원 논문의 YOLO v1은 이미지를 한 번의 forward pass로 처리한다.
    따라서 backbone CNN은 이미지 전체에서 spatial feature를 추출하고,
    head는 이를 S x S grid prediction으로 변환한다.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyYOLOv1(nn.Module):
    """
    YOLO v1의 핵심 출력 구조를 유지한 lightweight model이다.

    YOLO v1의 핵심은 detection 문제를 regression 문제로 변환하는 것이다.
    이미지를 S x S grid로 나누고, 각 grid cell이 다음 값을 한 번에 예측한다.

    1. B개의 bounding boxes
       각 bbox는 x, y, w, h, confidence를 가진다.

    2. C개의 class probability
       해당 cell에 객체 중심이 있을 때 class distribution을 예측한다.

    최종 output shape:
    [batch, S, S, B * 5 + C]
    """

    def __init__(self, s=7, b=2, c=3):
        super().__init__()

        self.s = s
        self.b = b
        self.c = c

        self.features = nn.Sequential(
            ConvBlock(3, 16, 3),
            nn.MaxPool2d(2, 2),

            ConvBlock(16, 32, 3),
            nn.MaxPool2d(2, 2),

            ConvBlock(32, 64, 3),
            nn.MaxPool2d(2, 2),

            ConvBlock(64, 128, 3),
            nn.MaxPool2d(2, 2),

            ConvBlock(128, 256, 3),
            ConvBlock(256, 256, 3),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * s * s, 1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, s * s * (b * 5 + c)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        x = x.view(-1, self.s, self.s, self.b * 5 + self.c)
        return x


def xywh_to_xyxy(box):
    """
    [center_x, center_y, width, height]를 [x1, y1, x2, y2]로 변환한다.
    IoU 계산을 위해 사용한다.
    """
    cx, cy, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]

    return torch.stack(
        [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
        ],
        dim=-1,
    )


def bbox_iou_xywh(box1, box2, eps=1e-6):
    """
    두 bounding box의 IoU를 계산한다.

    YOLO v1에서 confidence는 단순히 객체 존재 여부만 의미하지 않는다.
    논문에서는 confidence를 Pr(Object) * IoU로 정의한다.
    즉, box 안에 객체가 있을 확률과 box localization quality를 함께 반영한다.
    """

    b1 = xywh_to_xyxy(box1)
    b2 = xywh_to_xyxy(box2)

    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0)
    area2 = (b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0)

    return inter_area / (area1 + area2 - inter_area + eps)


class YOLOv1Loss(nn.Module):
    """
    YOLO v1 style loss다.

    loss는 크게 세 부분으로 나뉜다.

    1. coordinate loss
       객체가 있는 cell에서 responsible predictor의 x, y, w, h를 학습한다.
       원 논문처럼 w, h에는 sqrt를 적용한다.

    2. confidence loss
       responsible bbox는 IoU를 confidence target으로 사용한다.
       객체가 없는 bbox는 confidence target을 0으로 둔다.

    3. classification loss
       객체가 있는 cell에서 class probability를 학습한다.

    lambda_coord는 bbox 좌표 학습을 강조한다.
    lambda_noobj는 대부분의 cell이 배경인 detection 문제에서 no-object loss가 과도하게 커지는 것을 막는다.
    """

    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred, target):
        raise NotImplementedError(
            "전체 loss 구현은 Colab pipeline 코드에 포함되어 있다. "
            "이 파일은 구조 설명용이다."
        )
