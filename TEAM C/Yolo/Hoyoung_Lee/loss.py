import torch as th
import torch.nn as nn

class YOLO_Loss(nn.Module):
    def __init__(self, S, C, B, D, L_coord, L_noobj):
        super(YOLO_Loss, self).__init__()
        # L_coord: 위치 예측 에러에 가중치를 부여 (분류보다 위치를 더 중요하게 학습)
        # L_noobj: 배경(객체가 없는 셀)의 에러 가중치를 낮춤 (배경이 압도적으로 많기 때문)
        # ...

    def forward(self, y_pred, y_gt):
        # 1. 실제 정답 박스와 모델이 예측한 B개의 박스 간의 IOU(교집합 비율) 계산
        # 2. IOU가 가장 높은 박스 하나만을 '담당 박스(Responsible Box)'로 지정

        # --- A. 위치 손실 (Localization Loss) ---
        # 객체가 존재하는 셀에서, 담당 박스의 중심 좌표(x, y)와 너비/높이(w, h) 오차 계산
        # (너비/높이는 큰 박스와 작은 박스의 오차 비율을 맞추기 위해 제곱근(sqrt)을 씌워 계산)
        localization_loss = localization_center_loss + localization_dims_loss

        # --- B. 객체 유무 손실 (Objectness Loss) ---
        # 객체가 있는 셀(obj)과 없는 셀(noobj)의 Confidence Score(신뢰도) 예측 오차 계산
        objectness_loss = objectness_obj_loss + objectness_noobj_loss

        # --- C. 분류 손실 (Classification Loss) ---
        # 객체가 존재하는 셀에서 클래스 확률 예측 오차 계산
        classification_loss = th.sum(exists_obj_i * (gt_bboxes_class - pred_bboxes_class) ** 2)

        # 총 손실 = 위치 + 객체유무 + 분류
        total_loss = (localization_loss + objectness_loss + classification_loss) / n
        return total_loss