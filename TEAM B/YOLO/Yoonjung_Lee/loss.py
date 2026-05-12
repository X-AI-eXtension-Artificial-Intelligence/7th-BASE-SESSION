import copy
import torch as th
import torch.nn as nn
from torch.nn.functional import one_hot

# 1. 좌표 변환 함수: 중심점 형식(x, y, w, h)을 꼭짓점 형식(xmin, ymin, xmax, ymax)으로 변환한다.
# IOU를 계산하기 위해서는 상자의 경계면 좌표가 필요하기 때문에 반드시 거쳐야 하는 과정이다.
def get_bb_corners(bboxes_coords: th.Tensor) -> th.Tensor:
    """
    YOLO 형식의 중심점 좌표를 사용하여 상자의 네 꼭짓점 좌표를 계산한다.
    """
    # x_center에서 너비의 절반을 빼고 더하여 가로 경계를 구한다.
    xmin = bboxes_coords[..., 0] - bboxes_coords[..., 2] / 2
    # y_center에서 높이의 절반을 빼고 더하여 세로 경계를 구한다.
    ymin = bboxes_coords[..., 1] - bboxes_coords[..., 3] / 2
    xmax = bboxes_coords[..., 0] + bboxes_coords[..., 2] / 2
    ymax = bboxes_coords[..., 1] + bboxes_coords[..., 3] / 2

    # 계산된 네 좌표를 마지막 차원으로 쌓아서 반환한다.
    bb_corners = th.stack([xmin, ymin, xmax, ymax], dim=-1)
    return bb_corners

# 2. IOU 계산 함수: 두 상자 집합 사이의 교집합 면적 대비 합집합 면적의 비율을 구한다.
# 이 값이 1에 가까울수록 모델의 예측 위치가 정답과 일치함을 의미한다.
def iou(bboxes1_coords: th.Tensor, bboxes2_coords: th.Tensor) -> th.Tensor:
    """
    두 경계 상자 사이의 Intersection over Union 점수를 계산한다.
    """
    # 두 상자가 겹치는 영역(Intersection)의 왼쪽 위와 오른쪽 아래 좌표를 찾는다.
    xmin = th.max(bboxes1_coords[..., 0], bboxes2_coords[..., 0])
    ymin = th.max(bboxes1_coords[..., 1], bboxes2_coords[..., 1])
    xmax = th.min(bboxes1_coords[..., 2], bboxes2_coords[..., 2])
    ymax = th.min(bboxes1_coords[..., 3], bboxes2_coords[..., 3])
    
    # 각 상자의 전체 면적을 구한다 (가로 * 세로).
    area_bb1 = (bboxes1_coords[..., 2] - bboxes1_coords[..., 0]) * (bboxes1_coords[..., 3] - bboxes1_coords[..., 1])
    area_bb2 = (bboxes2_coords[..., 2] - bboxes2_coords[..., 0]) * (bboxes2_coords[..., 3] - bboxes2_coords[..., 1])

    # 겹치는 영역의 면적을 구한다. 아예 겹치지 않아 가로나 세로가 음수가 되는 경우는 0으로 처리한다.
    intersection = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
    # 합집합 면적 = 면적1 + 면적2 - 교집합면적 (중복 제거).
    union = area_bb1 + area_bb2 - intersection

    # 교집합을 합집합으로 나눈다. 1e-6은 분모가 0이 되어 연산 에러가 발생하는 것을 방지한다.
    return intersection / (union + 1e-6)

# 3. YOLO 손실 함수 클래스 정의
class YOLO_Loss(nn.Module):
    """
    YOLOv1의 3가지 손실(위치, 신뢰도, 분류)을 합산하여 최종 오차를 산출한다.
    """
    def __init__(self, S, C, B, D, L_coord, L_noobj):
        """
        초기화 단계에서 격자 크기(S), 클래스 수(C), 상자 수(B) 및 가중치 파라미터를 설정한다.
        """
        super(YOLO_Loss, self).__init__()
        self.S = S  # 격자 수 (7x7)
        self.B = B  # 격자당 상자 수 (2)
        self.C = C  # 클래스 수 (20)
        self.D = D  # 입력 이미지 해상도 (448)
        self.L_coord = L_coord # 위치 오차 가중치 (논문 권장 5.0)
        self.L_noobj = L_noobj # 배경 오차 가중치 (논문 권장 0.5)

        # 각 격자 내 B개의 상자 데이터가 시작되는 인덱스를 미리 계산하여 버퍼에 등록한다.
        self.register_buffer('pred_bb_ind', th.arange(start=self.C, end=self.C + self.B * 5).reshape(self.B, 5))

    def forward(self, y_pred, y_gt):
        """
        실제 손실 값을 계산하는 핵심 순전파 로직이다.
        """
        n = y_pred.shape[0] # 배치 크기
        # 1. 정답 데이터에서 실제로 물체가 존재하는 셀의 마스크를 가져온다.
        exists_obj_i = y_gt[..., 0:1] 
        # 2. 정답 좌표와 모델이 예측한 상자들의 좌표(루트 씌워진 형태)를 추출한다.
        gt_bboxes_coords = y_gt[..., None, self.C + 1:]
        pred_bboxes_sqrt_coords = y_pred[..., self.pred_bb_ind[:, 1:]]

        # 3. IOU 계산을 위해 좌표의 스케일을 원래 이미지 크기에 맞춰 복원하고 꼭짓점 형식으로 바꾼다.
        gt_bboxes_scaled_coords = copy.deepcopy(gt_bboxes_coords.data)
        gt_bboxes_scaled_coords[..., :2] /= self.S # 중심점을 전체 이미지 대비 비율로 변환
        gt_bboxes_coords_corners = get_bb_corners(gt_bboxes_scaled_coords)

        pred_bboxes_scaled_coords = copy.deepcopy(pred_bboxes_sqrt_coords.data)
        pred_bboxes_scaled_coords[..., :2] /= self.S 
        # 모델은 루트(w, h)를 예측하므로 다시 제곱하여 원래 크기 비율로 돌린다.
        pred_bboxes_scaled_coords[..., 2:] *= pred_bboxes_scaled_coords[..., 2:] 
        pred_bboxes_coords_corners = get_bb_corners(pred_bboxes_scaled_coords)

        # 4. 각 격자의 B개 상자 중 정답과 가장 많이 겹치는 '책임 상자'를 선정한다.
        iou_scores = iou(gt_bboxes_coords_corners, pred_bboxes_coords_corners)
        max_iou_score, max_iou_index = th.max(iou_scores, dim=-1)

        # IOU가 0인 경우(전혀 안 겹칠 때)를 대비하여 단순 거리(RMSE)가 가장 가까운 상자를 백업으로 찾는다.
        rmse_scores = th.sqrt(th.sum((gt_bboxes_scaled_coords - pred_bboxes_scaled_coords) ** 2, dim=-1))
        min_rmse_scores, min_rmse_index = th.min(rmse_scores, dim=-1)
        rmse_mask = max_iou_score == 0

        # 최종적으로 학습을 담당할 상자의 인덱스를 결정하고 원핫 인코딩 마스크를 만든다.
        best_index = max_iou_index
        best_index[rmse_mask] = min_rmse_index[rmse_mask]
        is_best_box = one_hot(best_index, self.B)

        # 물체가 있고 + 책임 상자로 뽑힌 경우에만 위치 오차를 계산하기 위한 마스크이다.
        exists_obj_ij = exists_obj_i * is_best_box
        # 배경 오차(물체가 없거나 책임 상자가 아닌 경우) 마스크이다.
        exists_noobj_ij = 1 - exists_obj_ij

        # 5. Localization Loss (위치 오차) 계산
        # 중심점(x, y)의 오차에 L_coord 가중치를 곱하여 합산한다.
        localization_center_loss = self.L_coord * th.sum(exists_obj_ij[..., None] * (
                (gt_bboxes_coords[..., 0:2] - pred_bboxes_sqrt_coords[..., 0:2]) ** 2))

        # 너비와 높이는 루트 값의 차이를 제곱하여 계산한다.
        localization_dims_loss = self.L_coord * th.sum(exists_obj_ij[..., None] * (
                (th.sqrt(gt_bboxes_coords[..., 2:4]) - pred_bboxes_sqrt_coords[..., 2:4]) ** 2))

        localization_loss = localization_center_loss + localization_dims_loss

        # 6. Objectness Loss (신뢰도 오차) 계산
        pred_bbox_cscores = y_pred[..., self.pred_bb_ind[:, 0]]

        # 물체가 있는 경우: 예측 신뢰도가 실제 IOU 점수와 같아지도록 유도한다.
        objectness_obj_loss = th.sum(exists_obj_ij * (iou_scores - pred_bbox_cscores) ** 2)
        # 물체가 없는 경우: 예측 신뢰도가 0이 되도록 하며 L_noobj 가중치를 곱한다.
        objectness_noobj_loss = self.L_noobj * th.sum(exists_noobj_ij * pred_bbox_cscores ** 2)

        objectness_loss = objectness_obj_loss + objectness_noobj_loss

        # 7. Classification Loss (분류 오차) 계산
        pred_bboxes_class = y_pred[..., :self.C]
        gt_bboxes_class = y_gt[..., 1:self.C + 1]

        # 물체가 존재하는 격자 셀에서만 20개 클래스 확률 오차를 구한다.
        classification_loss = th.sum(exists_obj_i * (gt_bboxes_class - pred_bboxes_class) ** 2)

        # 8. 최종 통합 손실 계산: 세 종류의 오차를 더하고 데이터 개수로 나누어 평균을 낸다.
        total_loss = (localization_loss + objectness_loss + classification_loss) / n
        return total_loss