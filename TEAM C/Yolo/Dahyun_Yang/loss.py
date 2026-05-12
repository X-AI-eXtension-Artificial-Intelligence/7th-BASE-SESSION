import copy

import torch as th
import torch.nn as nn
from torch.nn.functional import one_hot


def get_bb_corners(bboxes_coords: th.Tensor) -> th.Tensor:
    """
    중심 좌표 형식의 bounding box를 corner 좌표 형식으로 변환하는 함수.

    입력:
        (x_center, y_center, width, height)

    출력:
        (xmin, ymin, xmax, ymax)
    """

    xmin = bboxes_coords[..., 0] - bboxes_coords[..., 2] / 2
    ymin = bboxes_coords[..., 1] - bboxes_coords[..., 3] / 2
    xmax = bboxes_coords[..., 0] + bboxes_coords[..., 2] / 2
    ymax = bboxes_coords[..., 1] + bboxes_coords[..., 3] / 2

    bb_corners = th.stack([xmin, ymin, xmax, ymax], dim=-1)
    return bb_corners


def iou(bboxes1_coords: th.Tensor, bboxes2_coords: th.Tensor) -> th.Tensor:
    """
    두 bounding box 집합 사이의 IoU를 계산하는 함수.

    IoU = intersection area / union area

    입력 box 형식:
        (xmin, ymin, xmax, ymax)
    """
    # 두 box가 겹치는 영역의 왼쪽 위 좌표
    xmin = th.max(bboxes1_coords[..., 0], bboxes2_coords[..., 0])
    ymin = th.max(bboxes1_coords[..., 1], bboxes2_coords[..., 1])
    # 두 box가 겹치는 영역의 오른쪽 아래 좌표
    xmax = th.min(bboxes1_coords[..., 2], bboxes2_coords[..., 2])
    ymax = th.min(bboxes1_coords[..., 3], bboxes2_coords[..., 3])

    area_bb1 = (bboxes1_coords[..., 2] - bboxes1_coords[..., 0]) * (bboxes1_coords[..., 3] - bboxes1_coords[..., 1])
    area_bb2 = (bboxes2_coords[..., 2] - bboxes2_coords[..., 0]) * (bboxes2_coords[..., 3] - bboxes2_coords[..., 1])

    # clamp(min=0) for the special case: intersection=0
    # 겹치는 영역의 너비/높이가 음수가 될 수 있으므로 clamp(min=0)
    # 겹치지 않으면 intersection = 0
    intersection = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
    union = area_bb1 + area_bb2 - intersection

    # add 1e-6 to avoid division by 0
    return intersection / (union + 1e-6)


class YOLO_Loss(nn.Module):
    """
    YOLOv1 loss 함수 구현.

    YOLO loss는 크게 3가지로 구성됨.

    1. Localization loss
       - bounding box의 위치와 크기 오차

    2. Objectness loss
       - 해당 grid cell / bounding box가 object를 잘 예측했는지

    3. Classification loss
       - object가 있는 grid cell에서 class를 잘 맞혔는지
    """

    def __init__(self, S, C, B, D, L_coord, L_noobj):
        """
        S:
            이미지를 S x S grid로 나눔

        C:
            class 개수

        B:
            각 grid cell이 예측하는 bounding box 개수

        D:
            입력 이미지 크기
            예: 448이면 448 x 448

        L_coord:
            localization loss에 곱하는 가중치
            YOLOv1 논문에서는 보통 lambda_coord = 5

        L_noobj:
            object가 없는 box의 confidence loss에 곱하는 가중치
            YOLOv1 논문에서는 보통 lambda_noobj = 0.5
        """
        super(YOLO_Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.D = D
        self.L_coord = L_coord
        self.L_noobj = L_noobj

        # 각 bounding box 예측값의 index를 저장
        #
        # YOLO 출력 구조:
        # [class 정보 C개] + [bbox1 5개] + [bbox2 5개] + ...
        #
        # bbox 하나의 5개 값:
        # [confidence, x, y, w, h]

        # The indices for each of the B bounding boxes of the algorithm
        self.register_buffer('pred_bb_ind', th.arange(start=self.C, end=self.C + self.B * 5).reshape(self.B, 5))

    def forward(self, y_pred, y_gt):
        """
        YOLO loss 계산.

        y_pred shape:
            (N, S, S, C + B*5)

        y_gt shape:
            (N, S, S, C + 5)

        y_pred는 각 grid cell마다 class C개와 bbox B개를 예측함.
        y_gt는 각 grid cell마다 object 존재 여부, class one-hot, 정답 bbox 1개를 가짐.
        """
        # mini-batch 크기
        n = y_pred.shape[0]
        # object가 존재하는 grid cell이면 1, 아니면 0
        #
        # shape:
        # (N, S, S, 1)
        exists_obj_i = y_gt[..., 0:1]

        # ground truth bbox 좌표
        #
        # y_gt 구조:
        # [object_exists, class_one_hot..., x, y, w, h]
        #
        # self.C + 1부터 bbox 좌표
        #
        # None을 추가하는 이유:
        # pred bbox 개수 B와 비교하기 위해 차원을 맞춤
        #
        # shape:
        # (N, S, S, 1, 4)
        gt_bboxes_coords = y_gt[..., None, self.C + 1:]

        # 예측 bbox 좌표만 가져옴
        #
        # pred_bb_ind[:, 1:]은 각 bbox에서
        # confidence를 제외한 [x, y, w, h] index
        #
        # shape:
        # (N, S, S, B, 4)
        pred_bboxes_sqrt_coords = y_pred[..., self.pred_bb_ind[:, 1:]]

        # IoU 계산을 위해 ground truth 좌표를 복사
        # .data를 사용해 gradient 추적에서 분리
        gt_bboxes_scaled_coords = copy.deepcopy(gt_bboxes_coords.data)
        # x, y 좌표를 grid 기준으로 scale 조정
        gt_bboxes_scaled_coords[..., :2] /= self.S
        # 중심 좌표 형식 → corner 좌표 형식
        gt_bboxes_coords_corners = get_bb_corners(gt_bboxes_scaled_coords)

        # 예측 bbox 좌표도 IoU 계산용으로 복사
        pred_bboxes_scaled_coords = copy.deepcopy(pred_bboxes_sqrt_coords.data)
        # 예측 x, y도 scale 조정
        pred_bboxes_scaled_coords[..., :2] /= self.S
        # YOLOv1에서는 w, h의 sqrt를 예측하므로
        # IoU 계산을 위해 제곱해서 원래 w, h로 되돌림
        pred_bboxes_scaled_coords[..., 2:] *= pred_bboxes_scaled_coords[..., 2:]
        # 중심 좌표 형식 → corner 좌표 형식
        pred_bboxes_coords_corners = get_bb_corners(pred_bboxes_scaled_coords)


        # 각 grid cell에서 B개의 예측 bbox와 ground truth bbox 사이의 IoU 계산
        iou_scores = iou(gt_bboxes_coords_corners, pred_bboxes_coords_corners)
        # B개의 bbox 중 IoU가 가장 높은 bbox 선택
        max_iou_score, max_iou_index = th.max(iou_scores, dim=-1)

        rmse_scores = th.sqrt(th.sum((gt_bboxes_scaled_coords - pred_bboxes_scaled_coords) ** 2, dim=-1))
        min_rmse_scores, min_rmse_index = th.min(rmse_scores, dim=-1)
        rmse_mask = max_iou_score == 0

        # 기본적으로는 IoU가 가장 높은 bbox를 responsible box로 선택
        best_index = max_iou_index
        # IoU가 모두 0이면 RMSE가 가장 작은 bbox를 선택
        best_index[rmse_mask] = min_rmse_index[rmse_mask]
        # best bbox index를 one-hot으로 변환
        is_best_box = one_hot(best_index, self.B)

        exists_obj_ij = exists_obj_i * is_best_box
        exists_noobj_ij = 1 - exists_obj_ij

        # Localization Loss
        # 중심 좌표 x, y에 대한 loss
        # object가 있고 responsible bbox인 경우만 계산
        localization_center_loss = self.L_coord * th.sum(exists_obj_ij[..., None] * (
                (gt_bboxes_coords[..., 0:2] - pred_bboxes_sqrt_coords[..., 0:2]) ** 2))

        # gt에는 실제 w, h가 들어있으므로 sqrt를 씌워서
        # pred의 sqrt(w), sqrt(h)와 비교
        localization_dims_loss = self.L_coord * th.sum(exists_obj_ij[..., None] * (
                (th.sqrt(gt_bboxes_coords[..., 2:4]) - pred_bboxes_sqrt_coords[..., 2:4]) ** 2))
        # bbox의 위치와 크기를 맞히는 loss
        # YOLOv1은 width, height 자체가 아니라 sqrt(width), sqrt(height)를 예측
        localization_loss = localization_center_loss + localization_dims_loss

        # Objectness Loss
        # 예측 confidence score만 가져옴
        pred_bbox_cscores = y_pred[..., self.pred_bb_ind[:, 0]]
        # object가 있고 responsible bbox인 경우
        objectness_obj_loss = th.sum(exists_obj_ij * (iou_scores - pred_bbox_cscores) ** 2)
        objectness_noobj_loss = self.L_noobj * th.sum(exists_noobj_ij * pred_bbox_cscores ** 2) 
        # object가 없거나 responsible bbox가 아닌 경우    
        # 단순히 object가 있으면 1이 아니라, 예측 box가 정답 box와 얼마나 잘 겹치는지를 confidence target으로 사용
        objectness_loss = objectness_obj_loss + objectness_noobj_loss

        # Classification Loss
        pred_bboxes_class = y_pred[..., :self.C]
        gt_bboxes_class = y_gt[..., 1:self.C + 1]
        # object가 있는 grid cell에서만 classification loss를 계산
        classification_loss = th.sum(exists_obj_i * (gt_bboxes_class - pred_bboxes_class) ** 2)

        # Average YOLO Loss per instance
        total_loss = (localization_loss + objectness_loss + classification_loss) / n
        return total_loss
