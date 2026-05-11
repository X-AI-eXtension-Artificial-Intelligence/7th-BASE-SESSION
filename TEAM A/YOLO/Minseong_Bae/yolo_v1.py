"""
YOLOv1 implementation based on:
  "You Only Look Once: Unified, Real-Time Object Detection"
  Redmon et al., CVPR 2016
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ────────────────────────────────────────────────────────────
# Hyper-parameters (paper §2)
# ────────────────────────────────────────────────────────────
S = 7           # grid size
B = 2           # bounding boxes per cell
C = 20          # classes (PASCAL VOC)
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5


# ────────────────────────────────────────────────────────────
# Building block
# ────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Conv → Leaky ReLU (slope=0.1).  No BN — original paper pre-dates YOLOv2."""

    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=True)
        self.act  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


# ────────────────────────────────────────────────────────────
# Architecture  (Figure 3 of the paper)
# 24 convolutional layers + 2 fully-connected layers
# Input  : 448 × 448 × 3
# Output : S × S × (B*5 + C)  =  7 × 7 × 30
# ────────────────────────────────────────────────────────────
class YOLOv1(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S, self.B, self.C = S, B, C

        self.features = nn.Sequential(
            # Layer 1 — 448→224
            ConvBlock(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(2, 2),                             # 224→112

            # Layer 2
            ConvBlock(64, 192, 3, padding=1),
            nn.MaxPool2d(2, 2),                             # 112→56

            # Layers 3-6
            ConvBlock(192, 128, 1),
            ConvBlock(128, 256, 3, padding=1),
            ConvBlock(256, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            nn.MaxPool2d(2, 2),                             # 56→28

            # Layers 7-16  (×4 block: 1×1×256, 3×3×512)
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            # Layers 17-18
            ConvBlock(512, 512, 1),
            ConvBlock(512, 1024, 3, padding=1),
            nn.MaxPool2d(2, 2),                             # 28→14

            # Layers 19-22  (×2 block: 1×1×512, 3×3×1024)
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3, padding=1),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3, padding=1),
            # Layers 23-24
            ConvBlock(1024, 1024, 3, padding=1),
            ConvBlock(1024, 1024, 3, stride=2, padding=1),  # 14→7

            # Layers 25-26  (detection head conv layers)
            ConvBlock(1024, 1024, 3, padding=1),
            ConvBlock(1024, 1024, 3, padding=1),
        )

        # 2 fully-connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(S * S * 1024, 4096),
            nn.Dropout(0.5),                                # rate=0.5 (paper §2.2)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, S * S * (B * 5 + C)),          # linear activation (paper §2.2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.reshape(-1, self.S, self.S, self.B * 5 + self.C)


# ────────────────────────────────────────────────────────────
# IoU helper
# ────────────────────────────────────────────────────────────
def compute_iou(boxes_a, boxes_b):
    """
    boxes_a, boxes_b : (..., 4)  format (x_center, y_center, w, h)
    Returns IoU of matching boxes (element-wise).
    """
    ax1 = boxes_a[..., 0] - boxes_a[..., 2] / 2
    ay1 = boxes_a[..., 1] - boxes_a[..., 3] / 2
    ax2 = boxes_a[..., 0] + boxes_a[..., 2] / 2
    ay2 = boxes_a[..., 1] + boxes_a[..., 3] / 2

    bx1 = boxes_b[..., 0] - boxes_b[..., 2] / 2
    by1 = boxes_b[..., 1] - boxes_b[..., 3] / 2
    bx2 = boxes_b[..., 0] + boxes_b[..., 2] / 2
    by2 = boxes_b[..., 1] + boxes_b[..., 3] / 2

    ix1 = torch.max(ax1, bx1)
    iy1 = torch.max(ay1, by1)
    ix2 = torch.min(ax2, bx2)
    iy2 = torch.min(ay2, by2)

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    area_a = (ax2 - ax1).clamp(0) * (ay2 - ay1).clamp(0)
    area_b = (bx2 - bx1).clamp(0) * (by2 - by1).clamp(0)

    return inter / (area_a + area_b - inter + 1e-6)


# ────────────────────────────────────────────────────────────
# Loss function  (Equation 3 of the paper)
# ────────────────────────────────────────────────────────────
class YOLOv1Loss(nn.Module):
    """
    Multi-part MSE loss as defined in §2.1:

        λ_coord · Σ_ij^obj  [(x-x̂)² + (y-ŷ)²]
      + λ_coord · Σ_ij^obj  [(√w-√ŵ)² + (√h-√ĥ)²]
      +           Σ_ij^obj  (C-Ĉ)²
      + λ_noobj · Σ_ij^noobj (C-Ĉ)²
      +           Σ_i^obj   Σ_c (p(c)-p̂(c))²
    """

    def __init__(self, S=7, B=2, C=20,
                 lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S            = S
        self.B            = B
        self.C            = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        """
        predictions : (N, S, S, B*5 + C)
        targets     : (N, S, S, B*5 + C)

        Target encoding per cell (B=2, C=20):
          [x, y, w, h, conf,   x, y, w, h, conf,   cls_0 … cls_19]
           ←── box 0 ──────→   ←── box 1 ──────→   ←──── class ──→

        Both box slots carry the same ground-truth values; during
        loss computation only the "responsible" box (highest IoU)
        contributes to coordinate and object-confidence terms.
        """
        N = predictions.shape[0]

        pred = predictions.reshape(N, self.S, self.S, self.B, 5 + self.C // self.B
                                   if False else -1)   # keep flat for class slice
        # ── split into boxes and class probs ──────────────────────────
        pred_boxes = predictions[..., : self.B * 5].reshape(N, self.S, self.S, self.B, 5)
        pred_cls   = predictions[..., self.B * 5:]                            # (N,S,S,C)

        tgt_boxes  = targets[..., : self.B * 5].reshape(N, self.S, self.S, self.B, 5)
        tgt_cls    = targets[..., self.B * 5:]

        # obj_mask  : True where ground-truth object centre falls in cell
        obj_mask   = tgt_boxes[..., 0, 4] > 0          # (N, S, S)   use box-0 conf
        noobj_mask = ~obj_mask

        # ── find the responsible predictor (highest IoU with GT) ──────
        iou_per_box = torch.stack([
            compute_iou(pred_boxes[..., b, :4],
                        tgt_boxes[..., 0, :4])          # GT is same for all B slots
            for b in range(self.B)
        ], dim=-1)                                      # (N, S, S, B)

        best_box = iou_per_box.argmax(dim=-1, keepdim=True)   # (N, S, S, 1)
        resp     = torch.zeros_like(iou_per_box, dtype=torch.bool)
        resp.scatter_(-1, best_box, True)               # (N, S, S, B)

        # responsible AND has object
        obj_resp   = obj_mask.unsqueeze(-1) & resp      # (N, S, S, B)

        # ── coordinate loss: (x, y) ───────────────────────────────────
        xy_loss = (obj_resp * (
            (pred_boxes[..., 0] - tgt_boxes[..., 0]) ** 2
          + (pred_boxes[..., 1] - tgt_boxes[..., 1]) ** 2
        )).sum()

        # ── size loss: (√w, √h) — sign-safe sqrt ────────────────────
        pred_w = pred_boxes[..., 2].abs().sqrt() * pred_boxes[..., 2].sign()
        pred_h = pred_boxes[..., 3].abs().sqrt() * pred_boxes[..., 3].sign()
        tgt_w  = tgt_boxes[..., 2].sqrt()
        tgt_h  = tgt_boxes[..., 3].sqrt()

        wh_loss = (obj_resp * (
            (pred_w - tgt_w) ** 2
          + (pred_h - tgt_h) ** 2
        )).sum()

        # ── confidence loss (object cells) ────────────────────────────
        conf_obj_loss = (obj_resp * (
            pred_boxes[..., 4] - tgt_boxes[..., 4]
        ) ** 2).sum()

        # ── confidence loss (background cells) ────────────────────────
        noobj_mask_b = noobj_mask.unsqueeze(-1).expand_as(resp)
        conf_noobj_loss = (noobj_mask_b * (
            pred_boxes[..., 4] - tgt_boxes[..., 4]
        ) ** 2).sum()

        # ── class probability loss (object cells only) ────────────────
        cls_loss = (obj_mask.unsqueeze(-1) * (pred_cls - tgt_cls) ** 2).sum()

        # ── total loss (Eq. 3) ────────────────────────────────────────
        loss = (
            self.lambda_coord * (xy_loss + wh_loss)
          + conf_obj_loss
          + self.lambda_noobj * conf_noobj_loss
          + cls_loss
        ) / N

        return loss


# ────────────────────────────────────────────────────────────
# Dataset stub  (fill in for PASCAL VOC or custom data)
# ────────────────────────────────────────────────────────────
class VOCDataset(Dataset):
    """
    Placeholder.  Override __len__ and __getitem__ with real data.

    Expected __getitem__ output:
        image  : (3, 448, 448) float tensor
        target : (S, S, B*5+C) float tensor
    """

    def __init__(self, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.img_dir   = img_dir
        self.label_dir = label_dir
        self.S         = S
        self.B         = B
        self.C         = C
        self.transform = transform

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def build_target(self, boxes, class_ids):
        """
        Convert a list of bounding boxes into the YOLO target tensor.

        boxes      : list of [x_c, y_c, w, h]  (normalised 0-1, image-relative)
        class_ids  : list of int  (0-indexed class label)

        Returns    : (S, S, B*5 + C) float tensor
        """
        target = torch.zeros(self.S, self.S, self.B * 5 + self.C)

        for box, cls in zip(boxes, class_ids):
            x_c, y_c, w, h = box

            col = int(x_c * self.S)                     # grid column
            row = int(y_c * self.S)                     # grid row
            col = min(col, self.S - 1)
            row = min(row, self.S - 1)

            x_cell = x_c * self.S - col                 # relative to cell
            y_cell = y_c * self.S - row

            if target[row, col, 4] == 0:                # first object wins
                box_tensor = torch.tensor([x_cell, y_cell, w, h, 1.0])
                for b in range(self.B):
                    target[row, col, b * 5: b * 5 + 5] = box_tensor
                target[row, col, self.B * 5 + cls] = 1.0

        return target


# ────────────────────────────────────────────────────────────
# Learning-rate schedule  (paper §2.2)
# ────────────────────────────────────────────────────────────
def build_lr_scheduler(optimizer, warmup_epochs=1):
    """
    Warm-up  : 1e-3 → 1e-2  over first epoch
    Phase 1  : 1e-2  for 75 epochs
    Phase 2  : 1e-3  for 30 epochs
    Phase 3  : 1e-4  for 30 epochs
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # linear ramp from 0.1× to 1× of base lr
            return 0.1 + 0.9 * (epoch + 1) / warmup_epochs
        epoch -= warmup_epochs
        if epoch < 75:
            return 1.0
        if epoch < 75 + 30:
            return 0.1
        return 0.01

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, targets in loader:
        images  = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        loss  = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def train(
    model,
    train_loader,
    device,
    num_epochs=135,      # paper trains for ~135 epochs
    base_lr=1e-2,
):
    """
    Full training procedure from §2.2.

    SGD  momentum=0.9  weight_decay=5e-4  batch_size=64 (set in DataLoader)
    """
    model.to(device)
    criterion = YOLOv1Loss(S=S, B=B, C=C,
                           lambda_coord=LAMBDA_COORD,
                           lambda_noobj=LAMBDA_NOOBJ)
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = build_lr_scheduler(optimizer)

    for epoch in range(num_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch + 1:3d}/{num_epochs}]  loss: {loss:.4f}"
              f"  lr: {current_lr:.2e}")

    return model


# ────────────────────────────────────────────────────────────
# Sanity check
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = YOLOv1(S=S, B=B, C=C).to(device)

    # output shape check
    dummy = torch.randn(2, 3, 448, 448, device=device)
    out   = model(dummy)
    assert out.shape == (2, S, S, B * 5 + C), f"Unexpected shape: {out.shape}"
    print(f"Forward pass OK — output shape: {tuple(out.shape)}")  # (2, 7, 7, 30)

    # loss check
    criterion = YOLOv1Loss(S=S, B=B, C=C)
    fake_target = torch.zeros(2, S, S, B * 5 + C, device=device)
    loss = criterion(out, fake_target)
    print(f"Loss check OK  — loss: {loss.item():.4f}")

    # ── to run real training ──────────────────────────────────
    # transform = transforms.Compose([
    #     transforms.Resize((448, 448)),
    #     transforms.ToTensor(),
    # ])
    # dataset = VOCDataset("data/images", "data/labels", transform=transform)
    # loader  = DataLoader(dataset, batch_size=64, shuffle=True,
    #                      num_workers=4, pin_memory=True)
    # train(model, loader, device)
