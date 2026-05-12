import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from model import YOLOv1
from data import VOCDataset
from util import VOC_CLASSES

S, B, C = 7, 2, 20


# ─── YOLOv1 Loss (§2.3 in paper) ─────────────────────────────────────────────

class YOLOLoss(nn.Module):
    """
    YOLOv1 multi-part loss.

    pred / target shape: (N, S, S, B*5+C)

    Per-cell layout: [tx, ty, w, h, conf] × B  +  [p0…pC-1]

    Loss terms (λ_coord=5, λ_noobj=0.5):
      1. Coord loss  – MSE on (tx,ty) and (√w, √h) for responsible predictor
      2. Conf loss   – MSE on confidence for responsible predictor  (target=1)
      3. No-obj loss – λ_noobj × MSE on confidence for non-responsible / empty cells
      4. Class loss  – MSE on class probabilities for obj cells
    """

    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S            = S
        self.B            = B
        self.C            = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred, target):
        N      = pred.size(0)
        S, B, C = self.S, self.B, self.C
        device = pred.device

        # ── Masks ─────────────────────────────────────────────────────────
        obj_mask  = target[..., 4] > 0.5           # (N, S, S)  cells with object

        # ── Split tensors ─────────────────────────────────────────────────
        pred_boxes = pred[..., :B * 5].view(N, S, S, B, 5)   # (N,S,S,B,5)
        pred_cls   = pred[..., B * 5:]                         # (N,S,S,C)
        tgt_box    = target[..., :5]                           # (N,S,S,5)
        tgt_cls    = target[..., B * 5:]                       # (N,S,S,C)

        # ── Cell grid offsets (for IoU in image coords) ───────────────────
        rows = torch.arange(S, dtype=torch.float32, device=device)
        cols = torch.arange(S, dtype=torch.float32, device=device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')   # (S,S)
        grid_c = grid_c.unsqueeze(0).unsqueeze(-1)   # (1,S,S,1)
        grid_r = grid_r.unsqueeze(0).unsqueeze(-1)   # (1,S,S,1)

        # Predicted box → image-relative cx,cy (broadcast over B)
        cx_p = (grid_c + pred_boxes[..., 0]) / S    # (N,S,S,B)
        cy_p = (grid_r + pred_boxes[..., 1]) / S
        w_p  = pred_boxes[..., 2]
        h_p  = pred_boxes[..., 3]

        # GT box → image-relative cx,cy (broadcast over B via unsqueeze)
        cx_g = (grid_c + tgt_box[..., 0:1]) / S    # (N,S,S,1)
        cy_g = (grid_r + tgt_box[..., 1:2]) / S
        w_g  = tgt_box[..., 2:3]
        h_g  = tgt_box[..., 3:4]

        # ── IoU between each predicted box and its cell's GT box ──────────
        ix1   = torch.max(cx_p - w_p / 2, cx_g - w_g / 2)
        iy1   = torch.max(cy_p - h_p / 2, cy_g - h_g / 2)
        ix2   = torch.min(cx_p + w_p / 2, cx_g + w_g / 2)
        iy2   = torch.min(cy_p + h_p / 2, cy_g + h_g / 2)
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        ious  = inter / (w_p * h_p + w_g * h_g - inter + 1e-6)  # (N,S,S,B)

        # Responsible predictor: highest IoU per cell  →  (N,S,S,B) bool
        best_b   = ious.argmax(dim=-1, keepdim=True)             # (N,S,S,1)
        resp     = torch.zeros(N, S, S, B, dtype=torch.bool, device=device)
        resp.scatter_(-1, best_b, True)

        obj_b      = obj_mask.unsqueeze(-1).expand_as(resp)      # (N,S,S,B)
        resp_obj   = obj_b & resp                                  # responsible + has obj
        noobj_b    = ~obj_b                                        # cells with no object
        non_resp   = obj_b & ~resp                                 # obj cell, wrong predictor

        # ── 1. Coordinate loss ────────────────────────────────────────────
        coord_m = resp_obj.unsqueeze(-1).float()                   # (N,S,S,B,1)
        xy_pred = pred_boxes[..., :2]                              # (N,S,S,B,2)
        wh_pred = pred_boxes[..., 2:4]
        xy_tgt  = tgt_box[..., :2].unsqueeze(-2).expand_as(xy_pred)
        wh_tgt  = tgt_box[..., 2:4].unsqueeze(-2).expand_as(wh_pred)

        loss_xy = ((xy_pred - xy_tgt) ** 2 * coord_m).sum()
        # sqrt(w), sqrt(h) to reduce penalty for large box errors (§2.3)
        loss_wh = (
            (wh_pred.clamp(1e-6).sqrt() - wh_tgt.clamp(1e-6).sqrt()) ** 2
            * coord_m
        ).sum()
        loss_coord = self.lambda_coord * (loss_xy + loss_wh)

        # ── 2+3. Confidence loss ──────────────────────────────────────────
        conf_pred = pred_boxes[..., 4]                             # (N,S,S,B)

        # Responsible predictor in obj cell → target confidence = 1
        loss_conf_obj  = ((conf_pred - 1.0) ** 2 * resp_obj.float()).sum()

        # Non-responsible predictors + all noobj cells → target confidence = 0
        loss_conf_noobj = (
            conf_pred ** 2 * (noobj_b | non_resp).float()
        ).sum()
        loss_conf = loss_conf_obj + self.lambda_noobj * loss_conf_noobj

        # ── 4. Class loss ─────────────────────────────────────────────────
        cls_m    = obj_mask.unsqueeze(-1).float()                  # (N,S,S,1)
        loss_cls = ((pred_cls - tgt_cls) ** 2 * cls_m).sum()

        return (loss_coord + loss_conf + loss_cls) / N


# ─── Training utilities ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss  = criterion(preds, targets)

        if not torch.isfinite(loss):
            print(f'  [skip] non-finite loss: {loss.item():.4f}')
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def build_lr_scheduler(optimizer):
    """
    Paper schedule (§2.2): slowly raise 1e-3→1e-2 for 1 warm-up epoch,
    then 1e-2 for 75 ep, 1e-3 for 30 ep, 1e-4 for 30 ep.
    Base lr passed to SGD is 1e-2; lambda multiplies that.
    """
    def lr_lambda(epoch):
        if epoch < 1:
            return 0.1      # 1e-3  warm-up
        elif epoch < 76:
            return 1.0      # 1e-2
        elif epoch < 106:
            return 0.1      # 1e-3
        else:
            return 0.01     # 1e-4
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv1 on Pascal VOC')
    parser.add_argument('--data_root',    default='./data')
    parser.add_argument('--epochs',       type=int,   default=135)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_workers',  type=int,   default=4)
    parser.add_argument('--save_dir',     default='./checkpoints')
    parser.add_argument('--resume',       default=None,
                        help='path to checkpoint to resume from')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')

    # VOC 2007 trainval + VOC 2012 trainval (standard split)
    train_set = ConcatDataset([
        VOCDataset(args.data_root, year='2007', image_set='trainval', augment=True),
        VOCDataset(args.data_root, year='2012', image_set='trainval', augment=True),
    ])
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f'Train samples: {len(train_set)}')

    model     = YOLOv1(S=S, B=B, C=C).to(device)
    criterion = YOLOLoss(S=S, B=B, C=C)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {ckpt["epoch"]}')

    for epoch in range(start_epoch, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f'[{epoch:03d}/{args.epochs}]  loss={loss:.4f}  lr={lr_now:.2e}')

        if epoch % 10 == 0 or epoch == args.epochs:
            path = os.path.join(args.save_dir, f'yolov1_epoch{epoch:03d}.pth')
            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss':      loss,
            }, path)
            print(f'  → Saved {path}')


if __name__ == '__main__':
    main()
