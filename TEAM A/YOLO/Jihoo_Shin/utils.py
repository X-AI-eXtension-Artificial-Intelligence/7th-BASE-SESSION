"""
utils.py  ─  IoU, NMS, 예측값 디코딩, 시각화 헬퍼
"""

import torch
import numpy as np
import cv2
from typing import List, Tuple


# ──────────────────────────────────────────────────────────────
# IoU (x, y, w, h  형식)
# ──────────────────────────────────────────────────────────────
def iou_xywh(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    box1, box2: (..., 4)  형식 [cx, cy, w, h]  (0~1 정규화)
    반환: (...,) IoU
    """
    def to_xyxy(b):
        return torch.cat([b[..., :2] - b[..., 2:] / 2,
                          b[..., :2] + b[..., 2:] / 2], dim=-1)

    b1 = to_xyxy(box1)
    b2 = to_xyxy(box2)

    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union


# ──────────────────────────────────────────────────────────────
# 예측 텐서 → 박스 리스트 디코딩
# ──────────────────────────────────────────────────────────────
def decode_predictions(
    output: torch.Tensor,
    S: int = 7,
    B: int = 2,
    C: int = 20,
    conf_thresh: float = 0.25,
    img_size: int = 448,
) -> List[List[Tuple]]:
    """
    output : (N, S, S, C + B*5)
    반환   : N개의 이미지별 [(x1,y1,x2,y2, score, class_id), ...]  (픽셀 좌표)
    """
    N = output.shape[0]
    results = []

    for n in range(N):
        boxes = []
        pred = output[n]  # (S, S, C+B*5)

        for i in range(S):
            for j in range(S):
                cell = pred[i, j]
                class_probs = cell[:C]            # (C,)
                best_cls = class_probs.argmax().item()
                cls_prob = class_probs[best_cls].item()

                for b in range(B):
                    offset = C + b * 5
                    conf = cell[offset].item()
                    score = conf * cls_prob
                    if score < conf_thresh:
                        continue

                    cx_cell = cell[offset + 1].item()   # 셀 내부 상대좌표
                    cy_cell = cell[offset + 2].item()
                    w_rel   = cell[offset + 3].item()   # 이미지 전체 대비
                    h_rel   = cell[offset + 4].item()

                    # 이미지 전체 기준 절대 좌표
                    cx = (j + cx_cell) / S * img_size
                    cy = (i + cy_cell) / S * img_size
                    w  = w_rel * img_size
                    h  = h_rel * img_size

                    x1 = max(0, cx - w / 2)
                    y1 = max(0, cy - h / 2)
                    x2 = min(img_size, cx + w / 2)
                    y2 = min(img_size, cy + h / 2)

                    boxes.append((x1, y1, x2, y2, score, best_cls))

        results.append(boxes)
    return results


# ──────────────────────────────────────────────────────────────
# Non-Maximum Suppression
# ──────────────────────────────────────────────────────────────
def nms(boxes: List[Tuple], iou_thresh: float = 0.5) -> List[Tuple]:
    """
    boxes : [(x1,y1,x2,y2, score, class_id), ...]
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []

    while boxes:
        chosen = boxes.pop(0)
        keep.append(chosen)
        boxes = [
            b for b in boxes
            if b[5] != chosen[5] or _iou_xyxy(chosen, b) < iou_thresh
        ]
    return keep


def _iou_xyxy(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


# ──────────────────────────────────────────────────────────────
# Ground-Truth → YOLO target 텐서 변환
# ──────────────────────────────────────────────────────────────
def encode_target(
    gt_boxes_xyxy: List[Tuple],   # [(x1,y1,x2,y2, class_id), ...]
    S: int = 7,
    C: int = 20,
    img_size: int = 448,
) -> torch.Tensor:
    """
    반환: (S, S, C+5)  형식의 타겟 텐서
          [..., :C]      = one-hot 클래스
          [..., C]       = objectness (0 or 1)
          [..., C+1:C+5] = cx_cell, cy_cell, w_rel, h_rel
    """
    target = torch.zeros(S, S, C + 5)

    for (x1, y1, x2, y2, cls_id) in gt_boxes_xyxy:
        cx = (x1 + x2) / 2 / img_size
        cy = (y1 + y2) / 2 / img_size
        w  = (x2 - x1) / img_size
        h  = (y2 - y1) / img_size

        col = int(cx * S)
        row = int(cy * S)
        col = min(col, S - 1)
        row = min(row, S - 1)

        if target[row, col, C] == 1:   # 이미 박스 있으면 스킵
            continue

        cx_cell = cx * S - col
        cy_cell = cy * S - row

        target[row, col, cls_id] = 1.0            # one-hot
        target[row, col, C]      = 1.0            # objectness
        target[row, col, C + 1] = cx_cell
        target[row, col, C + 2] = cy_cell
        target[row, col, C + 3] = w
        target[row, col, C + 4] = h

    return target


# ──────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

COLORS = np.random.randint(0, 255, size=(20, 3), dtype=np.uint8).tolist()


def draw_boxes(img_bgr: np.ndarray, boxes: List[Tuple], class_names=VOC_CLASSES) -> np.ndarray:
    img = img_bgr.copy()
    for (x1, y1, x2, y2, score, cls_id) in boxes:
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_names[cls_id]}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(x1), int(y1) - th - 4),
                      (int(x1) + tw, int(y1)), color, -1)
        cv2.putText(img, label, (int(x1), int(y1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img
