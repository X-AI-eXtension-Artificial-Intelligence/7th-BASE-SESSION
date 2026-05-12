"""
YOLOv1 Multi-part Loss
----------------------
총 손실 = λ_coord * box_loss + obj_loss + λ_noobj * noobj_loss + class_loss

- box_loss   : 책임 박스(responsible box)의 (x, y) MSE + (√w, √h) MSE
- obj_loss   : 책임 박스의 confidence(IoU) MSE
- noobj_loss : 객체가 없는 셀의 confidence MSE
- class_loss : 객체가 있는 셀의 클래스 확률 MSE
"""

import torch
import torch.nn as nn
from utils import iou_xywh


class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, predictions, targets):
        """
        predictions : (N, S, S, C + B*5)
        targets     : (N, S, S, C + 5)   ← B=1 로 ground-truth 저장
        """
        N = predictions.shape[0]
        S, B, C = self.S, self.B, self.C

        # ── 분리 ──────────────────────────────────────────────────────────────
        # 각 박스: [conf, x, y, w, h]
        pred_boxes = predictions[..., C:C + B * 5].reshape(N, S, S, B, 5)
        pred_cls   = predictions[..., :C]

        gt_conf = targets[..., C:C + 1]        # (N, S, S, 1)
        gt_box  = targets[..., C + 1:C + 5]    # (N, S, S, 4)  x,y,w,h
        gt_cls  = targets[..., :C]             # (N, S, S, C)

        obj_mask   = gt_conf.squeeze(-1).bool()   # (N, S, S)
        noobj_mask = ~obj_mask

        # ── 책임 박스 선택 (IoU 가장 높은 박스) ───────────────────────────────
        # iou: (N, S, S, B)
        ious = torch.stack([
            iou_xywh(pred_boxes[..., b, 1:], gt_box)
            for b in range(B)
        ], dim=-1)                              # (N, S, S, B)

        best_box_idx = ious.argmax(dim=-1, keepdim=True)  # (N, S, S, 1)
        # responsible box 선택
        best_mask = torch.zeros_like(ious).scatter_(-1, best_box_idx, 1).bool()
        # (N, S, S, B) → (N, S, S) 책임 박스만 True
        resp_conf = pred_boxes[best_mask].reshape(N, S, S, 5)   # (N, S, S, 5)

        # ── 1. Box Coordinate Loss ─────────────────────────────────────────────
        pred_xy = resp_conf[..., 1:3][obj_mask]   # (n_obj, 2)
        pred_wh = resp_conf[..., 3:5][obj_mask]   # (n_obj, 2)
        gt_xy   = gt_box[..., :2][obj_mask]
        gt_wh   = gt_box[..., 2:][obj_mask]

        xy_loss = self.mse(pred_xy, gt_xy)
        wh_loss = self.mse(
            torch.sign(pred_wh) * torch.sqrt(pred_wh.abs() + 1e-6),
            torch.sqrt(gt_wh.clamp(min=0) + 1e-6)
        )
        box_loss = self.lambda_coord * (xy_loss + wh_loss)

        # ── 2. Object Confidence Loss ──────────────────────────────────────────
        best_iou = ious[best_mask].reshape(N, S, S)   # (N, S, S)
        pred_obj_conf = resp_conf[..., 0]              # (N, S, S)
        obj_loss = self.mse(
            pred_obj_conf[obj_mask],
            best_iou[obj_mask]                         # target = IoU
        )

        # ── 3. No-Object Confidence Loss ──────────────────────────────────────
        # 모든 B개 박스의 confidence에 페널티
        noobj_loss = self.lambda_noobj * self.mse(
            pred_boxes[..., 0][noobj_mask.unsqueeze(-1).expand_as(pred_boxes[..., 0])],
            torch.zeros_like(pred_boxes[..., 0])[noobj_mask.unsqueeze(-1).expand_as(pred_boxes[..., 0])]
        )

        # ── 4. Class Loss ──────────────────────────────────────────────────────
        cls_loss = self.mse(pred_cls[obj_mask], gt_cls[obj_mask])

        total_loss = (box_loss + obj_loss + noobj_loss + cls_loss) / N
        return total_loss, {
            'box':   box_loss.item() / N,
            'obj':   obj_loss.item() / N,
            'noobj': noobj_loss.item() / N,
            'cls':   cls_loss.item() / N,
        }
