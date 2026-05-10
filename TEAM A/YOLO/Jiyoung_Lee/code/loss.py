import copy

import torch as th
import torch.nn as nn
from torch.nn.functional import one_hot


def get_bb_corners(bboxes_coords: th.Tensor) -> th.Tensor:
    """
    YOLO bbox 형식(center format)을
    corner format으로 변환

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


def iou(bboxes1_coords: th.Tensor,
        bboxes2_coords: th.Tensor) -> th.Tensor:
    """
    IOU 계산

    IOU =
        intersection / union
    """

    # intersection 영역 계산
    xmin = th.max(bboxes1_coords[..., 0], bboxes2_coords[..., 0])
    ymin = th.max(bboxes1_coords[..., 1], bboxes2_coords[..., 1])

    xmax = th.min(bboxes1_coords[..., 2], bboxes2_coords[..., 2])
    ymax = th.min(bboxes1_coords[..., 3], bboxes2_coords[..., 3])

    # bbox1 넓이
    area_bb1 = (
        (bboxes1_coords[..., 2] - bboxes1_coords[..., 0]) *
        (bboxes1_coords[..., 3] - bboxes1_coords[..., 1])
    )

    # bbox2 넓이
    area_bb2 = (
        (bboxes2_coords[..., 2] - bboxes2_coords[..., 0]) *
        (bboxes2_coords[..., 3] - bboxes2_coords[..., 1])
    )

    # 겹치는 영역
    # clamp(min=0):
    # intersection 없을 때 음수 방지
    intersection = (
        (xmax - xmin).clamp(min=0) *
        (ymax - ymin).clamp(min=0)
    )

    # union 계산
    union = area_bb1 + area_bb2 - intersection

    # division by zero 방지
    return intersection / (union + 1e-6)


class YOLO_Loss(nn.Module):
    """
    YOLOv1 Loss 구현

    총 3개 loss 구성:
        1. localization loss
        2. objectness loss
        3. classification loss
    """

    def __init__(self,
                 S,
                 C,
                 B,
                 D,
                 L_coord,
                 L_noobj):

        super(YOLO_Loss, self).__init__()

        # grid 크기
        self.S = S

        # bbox 개수
        self.B = B

        # class 개수
        self.C = C

        # image size
        self.D = D

        # localization loss 가중치
        self.L_coord = L_coord

        # no object loss 가중치
        self.L_noobj = L_noobj

        """
        pred_bb_ind 예시:

        C=20, B=2 이면

        [
            [20,21,22,23,24],
            [25,26,27,28,29]
        ]

        각 bbox의:
            [confidence, x, y, w, h]
        index를 저장
        """
        self.register_buffer(
            'pred_bb_ind',
            th.arange(
                start=self.C,
                end=self.C + self.B * 5
            ).reshape(self.B, 5)
        )

    def forward(self, y_pred, y_gt):
        """
        YOLO Loss 계산
        """

        # batch size
        n = y_pred.shape[0]

        # object 존재 여부
        exists_obj_i = y_gt[..., 0:1]

        # GT bbox 좌표
        gt_bboxes_coords = y_gt[..., None, self.C + 1:]

        # 예측 bbox 좌표
        pred_bboxes_sqrt_coords = y_pred[..., self.pred_bb_ind[:, 1:]]

        # ---------------------------
        # GT bbox scaling
        # ---------------------------

        gt_bboxes_scaled_coords = copy.deepcopy(gt_bboxes_coords.data)

        # center 좌표 normalize
        gt_bboxes_scaled_coords[..., :2] /= self.S

        # corner format 변환
        gt_bboxes_coords_corners = get_bb_corners(
            gt_bboxes_scaled_coords
        )

        # ---------------------------
        # prediction bbox scaling
        # ---------------------------

        pred_bboxes_scaled_coords = copy.deepcopy(
            pred_bboxes_sqrt_coords.data
        )

        pred_bboxes_scaled_coords[..., :2] /= self.S

        # width, height는 sqrt 상태라 다시 제곱
        pred_bboxes_scaled_coords[..., 2:] *= \
            pred_bboxes_scaled_coords[..., 2:]

        pred_bboxes_coords_corners = get_bb_corners(
            pred_bboxes_scaled_coords
        )

        # ---------------------------
        # IOU 계산
        # ---------------------------

        iou_scores = iou(
            gt_bboxes_coords_corners,
            pred_bboxes_coords_corners
        )

        # 가장 높은 IOU 선택
        max_iou_score, max_iou_index = th.max(
            iou_scores,
            dim=-1
        )

        # ---------------------------
        # RMSE fallback
        # ---------------------------

        rmse_scores = th.sqrt(
            th.sum(
                (gt_bboxes_scaled_coords -
                 pred_bboxes_scaled_coords) ** 2,
                dim=-1
            )
        )

        min_rmse_scores, min_rmse_index = th.min(
            rmse_scores,
            dim=-1
        )

        # IOU가 모두 0이면 RMSE 사용
        rmse_mask = max_iou_score == 0

        best_index = max_iou_index
        best_index[rmse_mask] = min_rmse_index[rmse_mask]

        # responsible bbox 선택
        is_best_box = one_hot(best_index, self.B)

        # object 존재하는 bbox
        exists_obj_ij = exists_obj_i * is_best_box

        # object 없는 bbox
        exists_noobj_ij = 1 - exists_obj_ij

        # ==========================
        # Localization Loss
        # ==========================

        # center 좌표 loss
        localization_center_loss = self.L_coord * th.sum(
            exists_obj_ij[..., None] *
            (
                (gt_bboxes_coords[..., 0:2] -
                 pred_bboxes_sqrt_coords[..., 0:2]) ** 2
            )
        )

        # width/height loss
        localization_dims_loss = self.L_coord * th.sum(
            exists_obj_ij[..., None] *
            (
                (th.sqrt(gt_bboxes_coords[..., 2:4]) -
                 pred_bboxes_sqrt_coords[..., 2:4]) ** 2
            )
        )

        localization_loss = (
            localization_center_loss +
            localization_dims_loss
        )

        # ==========================
        # Objectness Loss
        # ==========================

        pred_bbox_cscores = y_pred[..., self.pred_bb_ind[:, 0]]

        # object 존재
        objectness_obj_loss = th.sum(
            exists_obj_ij *
            (iou_scores - pred_bbox_cscores) ** 2
        )

        # object 없음
        objectness_noobj_loss = self.L_noobj * th.sum(
            exists_noobj_ij *
            pred_bbox_cscores ** 2
        )

        objectness_loss = (
            objectness_obj_loss +
            objectness_noobj_loss
        )

        # ==========================
        # Classification Loss
        # ==========================

        pred_bboxes_class = y_pred[..., :self.C]

        gt_bboxes_class = y_gt[..., 1:self.C + 1]

        classification_loss = th.sum(
            exists_obj_i *
            (gt_bboxes_class - pred_bboxes_class) ** 2
        )

        # ==========================
        # Total Loss
        # ==========================

        total_loss = (
            localization_loss +
            objectness_loss +
            classification_loss
        ) / n

        return total_loss