"""
loss.py - YOLOv1 손실 함수 구현

YOLO 손실 함수는 3가지 구성요소의 합으로 계산됩니다:
  1. 위치 손실 (Localization Loss): 바운딩 박스 좌표 오차
  2. 객체성 손실 (Objectness Loss): 신뢰도 점수 오차
  3. 분류 손실 (Classification Loss): 클래스 확률 오차

유틸리티 함수:
  - get_bb_corners: 중심 형식 → 코너 형식 좌표 변환
  - iou: 두 바운딩 박스 집합 간 IoU 계산
"""

import copy

import torch as th
import torch.nn as nn
from torch.nn.functional import one_hot


def get_bb_corners(bboxes_coords: th.Tensor) -> th.Tensor:
    """
    YOLO 형식(중심 좌표)을 코너 형식으로 변환합니다.
    (x_center, y_center, width, height) → (xmin, ymin, xmax, ymax)

    :param bboxes_coords: 중심 형식 바운딩 박스 좌표 텐서
    :return: 코너 형식 바운딩 박스 좌표 텐서
    """
    xmin = bboxes_coords[..., 0] - bboxes_coords[..., 2] / 2
    ymin = bboxes_coords[..., 1] - bboxes_coords[..., 3] / 2
    xmax = bboxes_coords[..., 0] + bboxes_coords[..., 2] / 2
    ymax = bboxes_coords[..., 1] + bboxes_coords[..., 3] / 2

    bb_corners = th.stack([xmin, ymin, xmax, ymax], dim=-1)
    return bb_corners


def iou(bboxes1_coords: th.Tensor, bboxes2_coords: th.Tensor) -> th.Tensor:
    """
    두 바운딩 박스 집합 간의 IoU(Intersection over Union)를 계산합니다.
    좌표는 코너 형식 (xmin, ymin, xmax, ymax)이어야 합니다.

    :param bboxes1_coords: 첫 번째 바운딩 박스 집합
    :param bboxes2_coords: 두 번째 바운딩 박스 집합
    :return: 대응하는 바운딩 박스 쌍의 IoU 점수 텐서
    """
    # 교집합 영역의 좌표 계산
    xmin = th.max(bboxes1_coords[..., 0], bboxes2_coords[..., 0])
    ymin = th.max(bboxes1_coords[..., 1], bboxes2_coords[..., 1])
    xmax = th.min(bboxes1_coords[..., 2], bboxes2_coords[..., 2])
    ymax = th.min(bboxes1_coords[..., 3], bboxes2_coords[..., 3])

    # 각 박스의 면적 계산
    area_bb1 = (bboxes1_coords[..., 2] - bboxes1_coords[..., 0]) * (bboxes1_coords[..., 3] - bboxes1_coords[..., 1])
    area_bb2 = (bboxes2_coords[..., 2] - bboxes2_coords[..., 0]) * (bboxes2_coords[..., 3] - bboxes2_coords[..., 1])

    # 교집합 면적 (겹치지 않으면 0)
    intersection = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
    # 합집합 면적
    union = area_bb1 + area_bb2 - intersection

    # 0으로 나누기 방지를 위해 1e-6 추가
    return intersection / (union + 1e-6)


class YOLO_Loss(nn.Module):
    """
    YOLOv1 손실 함수.

    손실 = λ_coord × 위치손실 + 객체성손실 + 분류손실

    핵심 설계:
      - λ_coord (L_coord): 위치 손실 가중치를 높여 바운딩 박스 정확도 강조
      - λ_noobj (L_noobj): 객체 없는 셀의 신뢰도 손실 가중치를 낮춰 학습 안정성 확보
      - 너비/높이는 제곱근 형태로 예측하여 큰 박스와 작은 박스의 오차를 균형있게 처리
    """

    def __init__(self, S, C, B, D, L_coord, L_noobj):
        """
        :param S: 그리드 크기 (S × S)
        :param C: 클래스 수
        :param B: 셀당 예측 바운딩 박스 수
        :param D: 입력 이미지 크기 (D × D)
        :param L_coord: 위치 손실 가중치 (논문: 5.0)
        :param L_noobj: 비객체 신뢰도 손실 가중치 (논문: 0.5)
        """
        super(YOLO_Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.D = D
        self.L_coord = L_coord
        self.L_noobj = L_noobj

        # B개 바운딩 박스 각각의 인덱스 (신뢰도, x, y, w, h)
        self.register_buffer('pred_bb_ind', th.arange(start=self.C, end=self.C + self.B * 5).reshape(self.B, 5))

    def forward(self, y_pred, y_gt):
        """
        YOLO 손실 계산.

        처리 과정:
        1. GT와 예측 바운딩 박스의 IoU를 계산하여 "책임 박스" 선정
        2. 위치 손실: 중심 좌표 오차 + 크기(√w, √h) 오차
        3. 객체성 손실: 객체 있는 셀(IoU 타겟) + 객체 없는 셀(0 타겟)
        4. 분류 손실: 클래스 확률 오차

        :param y_pred: 모델 예측 (N, S, S, C+B*5)
        :param y_gt: 정답 레이블 (N, S, S, C+5)
        :return: 미니배치 평균 YOLO 손실
        """
        n = y_pred.shape[0]
        # 객체 존재 마스크 (1 또는 0)
        exists_obj_i = y_gt[..., 0:1]
        # GT 바운딩 박스 좌표
        gt_bboxes_coords = y_gt[..., None, self.C + 1:]
        # 예측 바운딩 박스 좌표 (√w, √h 형태)
        pred_bboxes_sqrt_coords = y_pred[..., self.pred_bb_ind[:, 1:]]

        # IoU 계산을 위해 좌표를 원본 스케일로 복원
        gt_bboxes_scaled_coords = copy.deepcopy(gt_bboxes_coords.data)
        gt_bboxes_scaled_coords[..., :2] /= self.S  # 셀 정규화 해제
        gt_bboxes_coords_corners = get_bb_corners(gt_bboxes_scaled_coords)

        pred_bboxes_scaled_coords = copy.deepcopy(pred_bboxes_sqrt_coords.data)
        pred_bboxes_scaled_coords[..., :2] /= self.S
        pred_bboxes_scaled_coords[..., 2:] *= pred_bboxes_scaled_coords[..., 2:]  # √w → w 복원
        pred_bboxes_coords_corners = get_bb_corners(pred_bboxes_scaled_coords)

        # 각 예측 박스와 GT 간 IoU 계산
        iou_scores = iou(gt_bboxes_coords_corners, pred_bboxes_coords_corners)
        max_iou_score, max_iou_index = th.max(iou_scores, dim=-1)

        # IoU가 0인 경우 RMSE가 가장 작은 박스를 선택
        rmse_scores = th.sqrt(th.sum((gt_bboxes_scaled_coords - pred_bboxes_scaled_coords) ** 2, dim=-1))
        min_rmse_scores, min_rmse_index = th.min(rmse_scores, dim=-1)
        rmse_mask = max_iou_score == 0

        # 최종 "책임 박스" 결정 (IoU 기반, IoU=0이면 RMSE 기반)
        best_index = max_iou_index
        best_index[rmse_mask] = min_rmse_index[rmse_mask]
        is_best_box = one_hot(best_index, self.B)

        # 객체가 존재하고 책임 박스인 경우의 마스크
        exists_obj_ij = exists_obj_i * is_best_box
        # 그 외 모든 경우 (비책임 박스 또는 객체 없는 셀)
        exists_noobj_ij = 1 - exists_obj_ij

        # ===== 1. 위치 손실 (Localization Loss) =====
        # 중심 좌표 오차
        localization_center_loss = self.L_coord * th.sum(exists_obj_ij[..., None] * (
                (gt_bboxes_coords[..., 0:2] - pred_bboxes_sqrt_coords[..., 0:2]) ** 2))
        # 크기 오차 (GT에 √ 적용하여 예측값과 비교)
        localization_dims_loss = self.L_coord * th.sum(exists_obj_ij[..., None] * (
                (th.sqrt(gt_bboxes_coords[..., 2:4]) - pred_bboxes_sqrt_coords[..., 2:4]) ** 2))

        localization_loss = localization_center_loss + localization_dims_loss

        # ===== 2. 객체성 손실 (Objectness Loss) =====
        pred_bbox_cscores = y_pred[..., self.pred_bb_ind[:, 0]]  # 예측 신뢰도

        # 객체 있는 셀: 타겟 = IoU 점수
        objectness_obj_loss = th.sum(exists_obj_ij * (iou_scores - pred_bbox_cscores) ** 2)
        # 객체 없는 셀: 타겟 = 0, λ_noobj로 가중치 감소
        objectness_noobj_loss = self.L_noobj * th.sum(exists_noobj_ij * pred_bbox_cscores ** 2)

        objectness_loss = objectness_obj_loss + objectness_noobj_loss

        # ===== 3. 분류 손실 (Classification Loss) =====
        pred_bboxes_class = y_pred[..., :self.C]
        gt_bboxes_class = y_gt[..., 1:self.C + 1]

        classification_loss = th.sum(exists_obj_i * (gt_bboxes_class - pred_bboxes_class) ** 2)

        # 미니배치 평균 총 손실
        total_loss = (localization_loss + objectness_loss + classification_loss) / n
        return total_loss
