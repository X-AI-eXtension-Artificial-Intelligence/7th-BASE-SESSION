import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    """
    논문 Eq. (3)의 multi-part loss 구현

    출력 텐서 구조 (S=7, B=2, C=20):
    [..., 0:10] = B개 bounding box (x, y, w, h, conf) × B
    [..., 10:30] = C개 class probability
    """
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord    # 논문: λ_coord = 5
        self.lambda_noobj = lambda_noobj    # 논문: λ_noobj = 0.5
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        """
        pred:   (batch, S, S, B*5+C)
        target: (batch, S, S, B*5+C)
        """
        batch = pred.size(0)

        # 객체가 있는 셀(1_obj_i), 없는 셀(1_noobj_i) 마스크
        # target[..., 4]: 첫 번째 박스의 confidence = 객체 존재 여부
        obj_mask = target[..., 4] > 0       # (batch, S, S)
        noobj_mask = ~obj_mask

        # -------------------------------------------------------
        # 1. Bounding Box Coordinate Loss (객체 있는 셀만)
        # -------------------------------------------------------
        pred_box = pred[..., :5][obj_mask]   # (N, 5): x, y, w, h, conf
        tgt_box = target[..., :5][obj_mask]

        # x, y 좌표 loss
        loss_xy = self.mse(pred_box[:, :2], tgt_box[:, :2])

        # w, h는 √w, √h로 계산 (논문: 작은 박스의 오차를 크게 반영)
        loss_wh = self.mse(
            torch.sign(pred_box[:, 2:4]) * torch.sqrt(torch.abs(pred_box[:, 2:4]) + 1e-6),
            torch.sqrt(tgt_box[:, 2:4] + 1e-6)
        )

        # -------------------------------------------------------
        # 2. Confidence Loss
        # -------------------------------------------------------
        # 객체 있는 셀: confidence → 1
        pred_conf_obj = pred[..., 4][obj_mask]
        tgt_conf_obj = target[..., 4][obj_mask]
        loss_conf_obj = self.mse(pred_conf_obj, tgt_conf_obj)

        # 객체 없는 셀: confidence → 0
        pred_conf_noobj = pred[..., 4][noobj_mask]
        tgt_conf_noobj = target[..., 4][noobj_mask]
        loss_conf_noobj = self.mse(pred_conf_noobj, tgt_conf_noobj)

        # -------------------------------------------------------
        # 3. Class Probability Loss (객체 있는 셀만)
        # -------------------------------------------------------
        pred_class = pred[..., self.B * 5:][obj_mask]
        tgt_class = target[..., self.B * 5:][obj_mask]
        loss_class = self.mse(pred_class, tgt_class)

        # -------------------------------------------------------
        # 최종 loss (논문 Eq. 3)
        # -------------------------------------------------------
        loss = (
            self.lambda_coord * (loss_xy + loss_wh)
            + loss_conf_obj
            + self.lambda_noobj * loss_conf_noobj
            + loss_class
        ) / batch

        return loss