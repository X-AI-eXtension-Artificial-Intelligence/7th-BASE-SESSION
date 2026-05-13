import copy

import torch as th
import torch.nn as nn
from torch.nn.functional import one_hot


def get_bb_corners(bboxes_coords: th.Tensor) -> th.Tensor:
    """
    YOLO 형식의 바운딩 박스 좌표 (x_center, y_center, width, height) 를
    코너 형식 (xmin, ymin, xmax, ymax) 으로 변환한다.
    """
    # 중심 좌표와 크기로부터 모서리 좌표 계산
    xmin = bboxes_coords[..., 0] - bboxes_coords[..., 2] / 2  # 왼쪽 경계
    ymin = bboxes_coords[..., 1] - bboxes_coords[..., 3] / 2  # 위쪽 경계
    xmax = bboxes_coords[..., 0] + bboxes_coords[..., 2] / 2  # 오른쪽 경계
    ymax = bboxes_coords[..., 1] + bboxes_coords[..., 3] / 2  # 아래쪽 경계

    bb_corners = th.stack([xmin, ymin, xmax, ymax], dim=-1)
    return bb_corners


def iou(bboxes1_coords: th.Tensor, bboxes2_coords: th.Tensor) -> th.Tensor:
    """
    두 바운딩 박스 집합 간의 IoU(Intersection over Union) 를 계산한다.
    입력 좌표는 코너 형식 (xmin, ymin, xmax, ymax) 이어야 한다.

    IoU = 교집합 넓이 / 합집합 넓이
    """
    # 교집합 영역의 좌표 계산 (두 박스 중 더 안쪽 경계)
    xmin = th.max(bboxes1_coords[..., 0], bboxes2_coords[..., 0])
    ymin = th.max(bboxes1_coords[..., 1], bboxes2_coords[..., 1])
    xmax = th.min(bboxes1_coords[..., 2], bboxes2_coords[..., 2])
    ymax = th.min(bboxes1_coords[..., 3], bboxes2_coords[..., 3])

    # 각 박스의 넓이
    area_bb1 = (bboxes1_coords[..., 2] - bboxes1_coords[..., 0]) * (bboxes1_coords[..., 3] - bboxes1_coords[..., 1])
    area_bb2 = (bboxes2_coords[..., 2] - bboxes2_coords[..., 0]) * (bboxes2_coords[..., 3] - bboxes2_coords[..., 1])

    # 교집합 넓이 (겹치지 않으면 0)
    intersection = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
    # 합집합 = 두 박스 넓이 합 - 교집합
    union = area_bb1 + area_bb2 - intersection

    # 1e-6: 분모가 0이 되는 경우를 방지
    return intersection / (union + 1e-6)


class YOLO_Loss(nn.Module):
    """
    YOLO v1 손실 함수 구현.
    손실은 세 가지 항의 합으로 구성된다:
      1. Localization Loss  : 바운딩 박스의 위치/크기 오차
      2. Objectness Loss    : 객체 존재 여부에 대한 신뢰도(confidence) 오차
      3. Classification Loss: 객체 클래스 분류 오차
    """

    def __init__(self, S, C, B, D, L_coord, L_noobj):
        """
        S       : 그리드 크기 (S×S)
        C       : 클래스 수
        B       : 셀당 예측 박스 수
        D       : 입력 이미지 크기 (D×D)
        L_coord : 위치 손실 가중치 (논문: 5) — 위치 오차를 더 강조
        L_noobj : 객체 없는 셀의 confidence 손실 가중치 (논문: 0.5) — 배경 셀 억제
        """
        super(YOLO_Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.D = D
        self.L_coord = L_coord
        self.L_noobj = L_noobj

        # 각 바운딩 박스의 예측값 인덱스를 미리 계산 (shape: [B, 5])
        # pred_bb_ind[b] = [confidence_idx, x_idx, y_idx, w_idx, h_idx]
        self.register_buffer('pred_bb_ind', th.arange(start=self.C, end=self.C + self.B * 5).reshape(self.B, 5))

    def forward(self, y_pred, y_gt):
        """
        y_pred: (N, S, S, C + B*5) — 모델의 예측값
        y_gt  : (N, S, S, C + 5)   — 정답 레이블

        정답 레이블 구조:
          [0]       : 해당 셀에 객체가 있으면 1, 없으면 0
          [1:C+1]   : 클래스 원-핫 인코딩
          [C+1:C+5] : (x_center, y_center, width, height) — 정규화된 좌표

        예측값 구조 (박스 하나당):
          [C + b*5]     : confidence (objectness score)
          [C + b*5 + 1] : x_center
          [C + b*5 + 2] : y_center
          [C + b*5 + 3] : sqrt(width)   ← 수렴 안정을 위해 제곱근 예측
          [C + b*5 + 4] : sqrt(height)
        """
        n = y_pred.shape[0]  # 미니배치 크기

        # ── 정답 데이터 분리 ──────────────────────────────────────────────
        exists_obj_i = y_gt[..., 0:1]                  # 셀에 객체 존재 여부 (N,S,S,1)
        gt_bboxes_coords = y_gt[..., None, self.C + 1:]  # 정답 박스 좌표 (N,S,S,1,4)

        # ── 예측 박스 좌표 추출 ───────────────────────────────────────────
        # pred_bb_ind[:, 1:] → 각 박스의 좌표 인덱스 (x, y, w, h)
        pred_bboxes_sqrt_coords = y_pred[..., self.pred_bb_ind[:, 1:]]  # (N,S,S,B,4)

        # ── IoU 계산을 위한 좌표 스케일 복원 ────────────────────────────────
        # 정답 좌표: 셀 기준 정규화 → 이미지 기준으로 스케일 복원
        gt_bboxes_scaled_coords = copy.deepcopy(gt_bboxes_coords.data)
        gt_bboxes_scaled_coords[..., :2] /= self.S  # 중심 좌표: 셀 오프셋 → 이미지 비율
        gt_bboxes_coords_corners = get_bb_corners(gt_bboxes_scaled_coords)

        # 예측 좌표: sqrt(w/h) → w/h 복원 후 코너 형식으로 변환
        pred_bboxes_scaled_coords = copy.deepcopy(pred_bboxes_sqrt_coords.data)
        pred_bboxes_scaled_coords[..., :2] /= self.S
        pred_bboxes_scaled_coords[..., 2:] *= pred_bboxes_scaled_coords[..., 2:]  # 제곱 → 실제 크기
        pred_bboxes_coords_corners = get_bb_corners(pred_bboxes_scaled_coords)

        # ── Responsible Box 선택 ─────────────────────────────────────────
        # B개의 예측 박스 중 정답과 IoU가 가장 높은 박스가 해당 셀의 예측을 담당
        iou_scores = iou(gt_bboxes_coords_corners, pred_bboxes_coords_corners)  # (N,S,S,B)
        max_iou_score, max_iou_index = th.max(iou_scores, dim=-1)

        # IoU가 0인 경우(전혀 겹치지 않음): RMSE가 가장 작은 박스를 담당으로 선택
        rmse_scores = th.sqrt(th.sum((gt_bboxes_scaled_coords - pred_bboxes_scaled_coords) ** 2, dim=-1))
        min_rmse_scores, min_rmse_index = th.min(rmse_scores, dim=-1)
        rmse_mask = max_iou_score == 0  # IoU가 0인 셀

        best_index = max_iou_index
        best_index[rmse_mask] = min_rmse_index[rmse_mask]  # IoU=0인 셀은 RMSE 최소 박스 선택

        # 담당 박스를 원-핫으로 표현: exists_obj_ij[i,j,b]=1 → 셀(i,j)의 박스 b가 담당
        is_best_box = one_hot(best_index, self.B)        # (N,S,S,B)
        exists_obj_ij = exists_obj_i * is_best_box       # 객체가 있는 셀의 담당 박스만 1
        exists_noobj_ij = 1 - exists_obj_ij              # 나머지 모든 박스

        # ── 1. Localization Loss ──────────────────────────────────────────
        # 중심 좌표 오차 (x, y): 담당 박스의 중심 좌표 예측 오차
        localization_center_loss = self.L_coord * th.sum(
            exists_obj_ij[..., None] * (
                (gt_bboxes_coords[..., 0:2] - pred_bboxes_sqrt_coords[..., 0:2]) ** 2
            )
        )
        # 크기 오차 (w, h): sqrt를 직접 비교하여 gradient 발산 방지
        # → 작은 박스의 오차를 더 강하게 반영하는 효과
        localization_dims_loss = self.L_coord * th.sum(
            exists_obj_ij[..., None] * (
                (th.sqrt(gt_bboxes_coords[..., 2:4]) - pred_bboxes_sqrt_coords[..., 2:4]) ** 2
            )
        )
        localization_loss = localization_center_loss + localization_dims_loss

        # ── 2. Objectness Loss ───────────────────────────────────────────
        # 담당 박스의 confidence 목표값 = 실제 IoU
        # 비담당 박스의 confidence 목표값 = 0
        pred_bbox_cscores = y_pred[..., self.pred_bb_ind[:, 0]]  # 예측 confidence (N,S,S,B)
        objectness_obj_loss   = th.sum(exists_obj_ij   * (iou_scores - pred_bbox_cscores) ** 2)
        objectness_noobj_loss = self.L_noobj * th.sum(exists_noobj_ij * pred_bbox_cscores ** 2)
        objectness_loss = objectness_obj_loss + objectness_noobj_loss

        # ── 3. Classification Loss ───────────────────────────────────────
        # 객체가 있는 셀에서만 클래스 예측 오차를 계산
        pred_bboxes_class = y_pred[..., :self.C]          # 예측 클래스 확률 (N,S,S,C)
        gt_bboxes_class   = y_gt[..., 1:self.C + 1]      # 정답 클래스 원-핫 (N,S,S,C)
        classification_loss = th.sum(exists_obj_i * (gt_bboxes_class - pred_bboxes_class) ** 2)

        # ── 최종 손실: 세 항의 합을 배치 크기로 평균 ─────────────────────────
        total_loss = (localization_loss + objectness_loss + classification_loss) / n
        return total_loss
