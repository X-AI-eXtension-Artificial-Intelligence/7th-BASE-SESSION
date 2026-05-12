"""
loss.py 이해용 한국어 주석 버전

YOLOv1 loss의 핵심:
1. localization loss: 박스 위치 오차
2. objectness loss: 물체가 있는지 confidence 오차
3. classification loss: 클래스 예측 오차
"""

import torch as th
import torch.nn as nn
from torch.nn.functional import one_hot


def get_bb_corners(bboxes_coords):
    """
    중심점 형식의 box를 모서리 형식으로 바꿉니다.

    입력 형식:
        (x_center, y_center, width, height)

    출력 형식:
        (xmin, ymin, xmax, ymax)

    IoU 계산은 보통 모서리 형식이 편합니다.
    """
    xmin = bboxes_coords[..., 0] - bboxes_coords[..., 2] / 2
    ymin = bboxes_coords[..., 1] - bboxes_coords[..., 3] / 2
    xmax = bboxes_coords[..., 0] + bboxes_coords[..., 2] / 2
    ymax = bboxes_coords[..., 1] + bboxes_coords[..., 3] / 2
    return th.stack([xmin, ymin, xmax, ymax], dim=-1)


def iou(boxes1, boxes2):
    """
    IoU, Intersection over Union 계산.

    IoU = 두 박스가 겹치는 면적 / 두 박스의 합집합 면적

    값이 1에 가까울수록 두 박스가 거의 같은 위치이고,
    값이 0에 가까울수록 거의 겹치지 않습니다.
    """
    xmin = th.max(boxes1[..., 0], boxes2[..., 0])
    ymin = th.max(boxes1[..., 1], boxes2[..., 1])
    xmax = th.min(boxes1[..., 2], boxes2[..., 2])
    ymax = th.min(boxes1[..., 3], boxes2[..., 3])

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 겹치지 않는 경우 width나 height가 음수가 되므로 0으로 자릅니다.
    intersection = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
    union = area1 + area2 - intersection

    # 0으로 나누는 문제 방지
    return intersection / (union + 1e-6)


class YOLO_Loss(nn.Module):
    """
    YOLOv1 손실함수.

    입력:
        y_pred: 모델 출력, shape = (N, S, S, C + B*5)
        y_gt: 정답, shape = (N, S, S, C + 5)

    y_gt는 셀마다 박스 1개만 가지고 있습니다.
    y_pred는 셀마다 B개 박스를 예측합니다.
    그래서 어떤 예측 박스가 정답을 담당할지 골라야 합니다.
    """

    def __init__(self, S, C, B, D, L_coord, L_noobj):
        super().__init__()
        self.S = S
        self.C = C
        self.B = B
        self.D = D
        self.L_coord = L_coord      # 박스 위치 손실에 더 큰 가중치
        self.L_noobj = L_noobj      # 물체 없는 셀 confidence 손실은 작게 반영

        # 예측값에서 각 bounding box가 차지하는 index를 미리 만들어 둡니다.
        # 예: C=20, B=2라면
        # box1: [20,21,22,23,24]
        # box2: [25,26,27,28,29]
        self.register_buffer(
            'pred_bb_ind',
            th.arange(start=C, end=C + B * 5).reshape(B, 5)
        )

    def forward(self, y_pred, y_gt):
        n = y_pred.shape[0]

        # 정답 셀에 물체가 있으면 1, 없으면 0
        exists_obj_i = y_gt[..., 0:1]

        # 정답 박스 좌표만 가져옵니다.
        gt_boxes = y_gt[..., None, self.C + 1:]

        # 예측 박스들의 좌표만 가져옵니다.
        # confidence는 제외하고 x, y, w, h만 가져오는 구조입니다.
        pred_boxes = y_pred[..., self.pred_bb_ind[:, 1:]]

        # 책임 박스 선택:
        # 각 셀에서 B개의 예측 박스 중 정답 박스와 IoU가 가장 높은 박스를 고릅니다.
        # 이 박스만 localization loss와 objectness obj loss를 담당합니다.
        #
        # 원본 코드는 좌표 normalization을 되돌린 뒤 IoU를 계산합니다.
        # 여기서는 핵심 개념만 주석으로 설명합니다.
        best_box_index = None
        is_best_box = None

        # localization loss:
        # 물체가 있는 셀 + 책임 박스에 대해서만 x, y, w, h 오차를 계산합니다.
        # YOLOv1은 큰 박스와 작은 박스의 균형을 맞추기 위해 w, h에 sqrt를 씁니다.
        localization_loss = 0

        # objectness loss:
        # 물체가 있고 책임 박스인 경우 confidence가 IoU에 가까워지도록 학습합니다.
        # 물체가 없는 경우 confidence가 0에 가까워지도록 학습합니다.
        objectness_loss = 0

        # classification loss:
        # 물체가 있는 셀에서만 클래스 예측 오차를 계산합니다.
        classification_loss = 0

        total_loss = (localization_loss + objectness_loss + classification_loss) / n
        return total_loss
