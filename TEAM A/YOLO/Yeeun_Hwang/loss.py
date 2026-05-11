import copy

import torch as th
import torch.nn as nn
from torch.nn.functional import one_hot


def get_bb_corners(bboxes_coords: th.Tensor) -> th.Tensor:
    xmin = bboxes_coords[..., 0] - bboxes_coords[..., 2] / 2
    ymin = bboxes_coords[..., 1] - bboxes_coords[..., 3] / 2
    xmax = bboxes_coords[..., 0] + bboxes_coords[..., 2] / 2
    ymax = bboxes_coords[..., 1] + bboxes_coords[..., 3] / 2

    bb_corners = th.stack([xmin, ymin, xmax, ymax], dim=-1)
    return bb_corners


def iou(bboxes1_coords: th.Tensor, bboxes2_coords: th.Tensor) -> th.Tensor:
    """
    두 바운딩 박스 집합 간의 Intersection over Union(IOU) 계산.
    입력은 코너 포맷 (xmin, ymin, xmax, ymax) 이어야 합니다.

    [수식]
    - intersection = clamp(xmax_inter - xmin_inter, 0) * clamp(ymax_inter - ymin_inter, 0)
    - union = area1 + area2 - intersection
    - IOU = intersection / (union + 1e-6)    ← 0 나눗셈 방지용 epsilon

    [주의]
    - 두 입력 텐서의 shape이 브로드캐스팅 가능해야 합니다.
    - 음수 width/height를 갖는 박스가 들어오면 area 계산이 잘못됩니다.
      따라서 호출 전 유효한 좌표임을 보장해야 합니다.
    """
    # 교집합 영역의 코너 좌표
    xmin = th.max(bboxes1_coords[..., 0], bboxes2_coords[..., 0])
    ymin = th.max(bboxes1_coords[..., 1], bboxes2_coords[..., 1])
    xmax = th.min(bboxes1_coords[..., 2], bboxes2_coords[..., 2])
    ymax = th.min(bboxes1_coords[..., 3], bboxes2_coords[..., 3])

    area_bb1 = (bboxes1_coords[..., 2] - bboxes1_coords[..., 0]) * (bboxes1_coords[..., 3] - bboxes1_coords[..., 1])
    area_bb2 = (bboxes2_coords[..., 2] - bboxes2_coords[..., 0]) * (bboxes2_coords[..., 3] - bboxes2_coords[..., 1])

    # 교집합이 없는 경우(두 박스가 겹치지 않음) → clamp(min=0)으로 0 처리
    intersection = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
    union = area_bb1 + area_bb2 - intersection

    # 1e-6: union=0인 엣지 케이스(두 박스 모두 degenerate) 대비
    return intersection / (union + 1e-6)


class YOLO_Loss(nn.Module):
    """
    YOLOv1 논문의 Multi-part Loss 구현.

    손실 = L_coord * (loc_center + loc_dims) + obj_loss + cls_loss

    [하이퍼파라미터 역할]
    - L_coord (기본값 5.0) : 위치 손실 가중치 증가 → bbox 정확도 강조
    - L_noobj (기본값 0.5) : 객체 없는 셀의 confidence 손실 감소
                             → 배경이 많은 이미지에서 학습 불안정성 억제

    [텐서 포맷]
    - y_pred : (N, S, S, C + B*5)
               클래스 확률 C개 + 박스별 [conf, cx, cy, sw, sh] * B개
               (sw, sh = sqrt(w), sqrt(h))
    - y_gt   : (N, S, S, C + 5)
               [obj_flag, cls_one_hot × C, cx, cy, w, h]
    """

    def __init__(self, S, C, B, D, L_coord, L_noobj):
        super(YOLO_Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.D = D
        self.L_coord = L_coord
        self.L_noobj = L_noobj

        # pred_bb_ind[b] = b번째 박스의 [conf, cx, cy, sw, sh] 인덱스 배열
        # register_buffer: 모델 저장/로드 시 포함되지만 학습 파라미터는 아님
        self.register_buffer('pred_bb_ind', th.arange(start=self.C, end=self.C + self.B * 5).reshape(self.B, 5))

    def forward(self, y_pred, y_gt):
        """
        YOLO 손실 계산의 핵심 로직.

        [Responsible Box 선정 절차]
        1. B개 예측 박스 각각과 GT 박스 간 IOU 계산
        2. IOU 최대인 박스를 responsible box로 선정
        3. 만약 모든 박스의 IOU=0이면 RMSE 최소 박스 선정 (단, RMSE < 20 조건)
        4. 그 외 경우 랜덤 선정 (논문 구현 그대로 따름)

        [스케일 정규화 주의사항]
        - y_gt의 cx, cy는 셀 내 상대 좌표 [0,1] → /S 로 이미지 전체 좌표로 변환
        - y_pred의 sw, sh는 sqrt(w/D), sqrt(h/D) 형태 → 제곱 후 실제 비율 획득
        - IOU 계산 시에는 이 스케일 변환이 필수 (안 하면 IOU 부정확)
        - 손실 계산 시에는 원래 정규화 좌표 그대로 사용

        [copy.deepcopy 사용 이유]
        - 스케일 변환(in-place 연산)이 autograd 그래프를 오염시키지 않도록
          .data를 복사해 계산 그래프와 완전히 분리된 텐서에서 IOU 연산 수행

        """
        n = y_pred.shape[0]  # 배치 크기

        # exists_obj_i : (N, S, S, 1) — 셀에 객체가 있으면 1, 없으면 0
        exists_obj_i = y_gt[..., 0:1]
        # gt_bboxes_coords : (N, S, S, 1, 4) — GT 박스 (cx, cy, w, h)
        # None 차원 삽입으로 B개 예측 박스와 브로드캐스팅 가능하게 만듦
        gt_bboxes_coords = y_gt[..., None, self.C + 1:]
        # pred_bboxes_sqrt_coords : (N, S, S, B, 4) — 예측 박스 (cx, cy, sw, sh)
        pred_bboxes_sqrt_coords = y_pred[..., self.pred_bb_ind[:, 1:]]

        # --- IOU 계산을 위한 실제 이미지 좌표 복원 ---
        # [주의] deepcopy + .data : autograd 그래프 외부에서 좌표 변환 수행
        gt_bboxes_scaled_coords = copy.deepcopy(gt_bboxes_coords.data)
        gt_bboxes_scaled_coords[..., :2] /= self.S          # (cx,cy) → 이미지 비율
        gt_bboxes_coords_corners = get_bb_corners(gt_bboxes_scaled_coords)

        pred_bboxes_scaled_coords = copy.deepcopy(pred_bboxes_sqrt_coords.data)
        pred_bboxes_scaled_coords[..., :2] /= self.S        # (cx,cy) → 이미지 비율
        pred_bboxes_scaled_coords[..., 2:] *= pred_bboxes_scaled_coords[..., 2:]  # sqrt → 실제값
        pred_bboxes_coords_corners = get_bb_corners(pred_bboxes_scaled_coords)

        # --- Responsible Box 선정 ---
        # iou_scores : (N, S, S, B)
        iou_scores = iou(gt_bboxes_coords_corners, pred_bboxes_coords_corners)
        max_iou_score, max_iou_index = th.max(iou_scores, dim=-1)

        # IOU=0인 경우 RMSE 기반 fallback
        rmse_scores = th.sqrt(th.sum((gt_bboxes_scaled_coords - pred_bboxes_scaled_coords) ** 2, dim=-1))
        min_rmse_scores, min_rmse_index = th.min(rmse_scores, dim=-1)
        rmse_mask = max_iou_score == 0

        best_index = max_iou_index
        best_index[rmse_mask] = min_rmse_index[rmse_mask]

        # is_best_box : (N, S, S, B) — responsible box 위치에만 1
        is_best_box = one_hot(best_index, self.B)

        # exists_obj_ij  : (N, S, S, B) — 객체가 있고 responsible box인 경우만 1
        # exists_noobj_ij: (N, S, S, B) — 나머지 모든 경우 1
        exists_obj_ij   = exists_obj_i * is_best_box
        exists_noobj_ij = 1 - exists_obj_ij


        # 1. Localization Loss
        # 중심 좌표 오차 (cx, cy)
        localization_center_loss = self.L_coord * th.sum(exists_obj_ij[..., None] * (
                (gt_bboxes_coords[..., 0:2] - pred_bboxes_sqrt_coords[..., 0:2]) ** 2))

        # 크기 오차 (sqrt(w), sqrt(h)) — GT도 sqrt 취해 예측값과 동일 스케일 비교
        # [이유] w/h 직접 비교 시 sqrt(x)의 x=0 근방 gradient = +inf → 학습 발산 위험
        localization_dims_loss = self.L_coord * th.sum(exists_obj_ij[..., None] * (
                (th.sqrt(gt_bboxes_coords[..., 2:4]) - pred_bboxes_sqrt_coords[..., 2:4]) ** 2))

        localization_loss = localization_center_loss + localization_dims_loss

        # 2. Objectness Loss
        # 예측 confidence score : (N, S, S, B)
        pred_bbox_cscores = y_pred[..., self.pred_bb_ind[:, 0]]

        # 객체가 있는 셀: target confidence = IOU (예측 박스가 GT와 얼마나 겹치는지)
        objectness_obj_loss   = th.sum(exists_obj_ij * (iou_scores - pred_bbox_cscores) ** 2)
        # 객체 없는 셀: target confidence = 0, L_noobj로 스케일 다운
        objectness_noobj_loss = self.L_noobj * th.sum(exists_noobj_ij * pred_bbox_cscores ** 2)

        objectness_loss = objectness_obj_loss + objectness_noobj_loss

        # 3. Classification Loss
        # 객체가 있는 셀에 대해서만 클래스 예측 오차를 적용
        pred_bboxes_class = y_pred[..., :self.C]
        gt_bboxes_class   = y_gt[..., 1:self.C + 1]

        classification_loss = th.sum(exists_obj_i * (gt_bboxes_class - pred_bboxes_class) ** 2)

        # 총 손실: 배치 평균
        total_loss = (localization_loss + objectness_loss + classification_loss) / n
        return total_loss