import torch
import numpy as np

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]


# ─── IoU helpers ──────────────────────────────────────────────────────────────

def iou(box1, box2):
    """Scalar IoU. box format: [x1, y1, x2, y2] normalized to [0,1]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def iou_tensor(boxes1, boxes2):
    """
    Batched IoU.
    boxes1: (N, 4)  boxes2: (M, 4)  →  returns (N, M)
    Format: [x1, y1, x2, y2]
    """
    b1 = boxes1.unsqueeze(1)   # (N, 1, 4)
    b2 = boxes2.unsqueeze(0)   # (1, M, 4)

    ix1 = torch.max(b1[..., 0], b2[..., 0])
    iy1 = torch.max(b1[..., 1], b2[..., 1])
    ix2 = torch.min(b1[..., 2], b2[..., 2])
    iy2 = torch.min(b1[..., 3], b2[..., 3])

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = a1.unsqueeze(1) + a2.unsqueeze(0) - inter
    return inter / (union + 1e-6)


# ─── NMS ──────────────────────────────────────────────────────────────────────

def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes:  (N, 4) [x1, y1, x2, y2]
    scores: (N,)
    Returns list of kept indices sorted by score descending.
    """
    order = scores.argsort(descending=True)
    kept = []
    while order.numel() > 0:
        i = order[0].item()
        kept.append(i)
        if order.numel() == 1:
            break
        rest_ious = iou_tensor(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][rest_ious <= iou_threshold]
    return kept


# ─── Decode ───────────────────────────────────────────────────────────────────

def decode_predictions(output, S=7, B=2, C=20, conf_thresh=0.25, nms_thresh=0.5):
    """
    Convert model output to a list of detections for a single image.

    output: (S, S, B*5+C)  or  (1, S, S, B*5+C)
    Returns list of [x1, y1, x2, y2, score, class_idx] in [0,1] image coords.

    Each cell output layout: [tx, ty, w, h, conf] × B  +  [p0..pC-1]
      tx, ty : offset from cell top-left, normalized by cell size  ∈ [0,1]
      w,  h  : box dimensions relative to whole image              ∈ [0,1]
      conf   : objectness confidence
      score  = conf × class_probability
    """
    if output.dim() == 4:
        output = output[0]
    output = output.detach().cpu()

    all_boxes, all_scores, all_cls = [], [], []
    cell = 1.0 / S

    for row in range(S):
        for col in range(S):
            cell_pred = output[row, col]
            cls_probs = cell_pred[B * 5:]           # (C,)
            cls_idx   = cls_probs.argmax().item()
            cls_conf  = cls_probs[cls_idx].item()

            for b in range(B):
                base = b * 5
                conf  = cell_pred[base + 4].item()
                score = conf * cls_conf
                if score < conf_thresh:
                    continue

                tx = cell_pred[base + 0].item()
                ty = cell_pred[base + 1].item()
                w  = cell_pred[base + 2].item()
                h  = cell_pred[base + 3].item()

                cx = (col + tx) * cell
                cy = (row + ty) * cell
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(score)
                all_cls.append(cls_idx)

    if not all_boxes:
        return []

    boxes_t  = torch.tensor(all_boxes,  dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    cls_t    = torch.tensor(all_cls,    dtype=torch.int64)

    results = []
    for c in cls_t.unique():
        mask = cls_t == c
        kept = nms(boxes_t[mask], scores_t[mask], nms_thresh)
        for k in kept:
            b = boxes_t[mask][k].tolist()
            s = scores_t[mask][k].item()
            results.append(b + [s, c.item()])

    return results


# ─── Encode targets ───────────────────────────────────────────────────────────

def encode_targets(boxes, labels, S=7, B=2, C=20):
    """
    Convert GT boxes to the S×S×(B*5+C) target tensor expected by YOLOLoss.

    boxes:  list of [x1, y1, x2, y2] normalized to [0,1]
    labels: list of int class indices (same length as boxes)

    Target layout per cell: [tx, ty, w, h, conf] × B  +  one-hot class
      tx = cx*S - col   (x offset within cell)
      ty = cy*S - row   (y offset within cell)
      w, h              (image-relative box dimensions)
      conf = 1.0        (set to 1 for cells that own an object)

    The same GT box is written into all B predictor slots; YOLOLoss
    decides which predictor is "responsible" at training time via IoU.
    If two objects fall in the same cell, only the first is kept.
    """
    target = torch.zeros(S, S, B * 5 + C)

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1

        col = min(int(cx * S), S - 1)
        row = min(int(cy * S), S - 1)

        if target[row, col, 4] != 0:    # cell already claimed
            continue

        tx = cx * S - col
        ty = cy * S - row

        for b in range(B):
            base = b * 5
            target[row, col, base:base + 5] = torch.tensor(
                [tx, ty, w, h, 1.0], dtype=torch.float32
            )
        target[row, col, B * 5 + label] = 1.0

    return target
