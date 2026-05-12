import torch
import torch.nn as nn


# ─── Building block ───────────────────────────────────────────────────────────

def _conv(in_ch, out_ch, k, s=1, p=0):
    """Conv → BatchNorm → LeakyReLU(0.1).
    Original paper does not use BatchNorm, but it is added here for
    training stability on limited hardware.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


# ─── YOLOv1 ───────────────────────────────────────────────────────────────────

class YOLOv1(nn.Module):
    """
    YOLOv1 (Redmon et al., 2016) — Table 1 in the paper.

    Architecture: 24 conv layers → Flatten → FC(4096) → FC(S·S·(B·5+C))
    Output is reshaped to (N, S, S, B*5+C) before returning.

    Input:  (N, 3, 448, 448)
    Output: (N, S, S, B*5+C)   default → (N, 7, 7, 30)

    Per-cell output layout:
      [tx, ty, w, h, conf]  ×  B  boxes
      [p0 … pC-1]           class probabilities (shared across B boxes)
    """

    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

        self.features = nn.Sequential(
            # ── Block 1 ──────────────────────────── 448 → 224 → 112
            _conv(3,    64,  7, s=2, p=3),      # Conv 7×7/2,  64
            nn.MaxPool2d(2, 2),

            # ── Block 2 ──────────────────────────── 112 → 56
            _conv(64,  192,  3, p=1),            # Conv 3×3,   192
            nn.MaxPool2d(2, 2),

            # ── Block 3 ──────────────────────────── 56 (no pool)
            _conv(192, 128,  1),                 # Conv 1×1,   128
            _conv(128, 256,  3, p=1),            # Conv 3×3,   256
            _conv(256, 256,  1),                 # Conv 1×1,   256
            _conv(256, 512,  3, p=1),            # Conv 3×3,   512
            nn.MaxPool2d(2, 2),                  #             56 → 28

            # ── Block 4 ──────────────────────────── 28  (4× repeated pair)
            _conv(512, 256,  1),                 # ┐
            _conv(256, 512,  3, p=1),            # ┘ × 4
            _conv(512, 256,  1),
            _conv(256, 512,  3, p=1),
            _conv(512, 256,  1),
            _conv(256, 512,  3, p=1),
            _conv(512, 256,  1),
            _conv(256, 512,  3, p=1),
            _conv(512, 512,  1),                 # Conv 1×1,   512
            _conv(512, 1024, 3, p=1),            # Conv 3×3,  1024
            nn.MaxPool2d(2, 2),                  #             28 → 14

            # ── Block 5 ──────────────────────────── 14  (2× repeated pair)
            _conv(1024,  512, 1),                # ┐
            _conv(512,  1024, 3, p=1),           # ┘ × 2
            _conv(1024,  512, 1),
            _conv(512,  1024, 3, p=1),
            _conv(1024, 1024, 3, p=1),           # Conv 3×3,  1024
            _conv(1024, 1024, 3, s=2, p=1),      # Conv 3×3/2, 1024  → 7

            # ── Block 6 ──────────────────────────── 7
            _conv(1024, 1024, 3, p=1),           # Conv 3×3,  1024
            _conv(1024, 1024, 3, p=1),           # Conv 3×3,  1024
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        x = self.features(x)                            # (N, 1024, S, S)
        x = self.classifier(x)                          # (N, S·S·(B*5+C))
        x = x.view(-1, self.S, self.S,
                   self.B * 5 + self.C)                 # (N, S, S, B*5+C)
        # sigmoid로 모든 출력을 (0,1)에 고정 → tx,ty,w,h,conf,class 전부 bounded
        # 이로 인해 w,h가 음수가 되거나 conf가 폭발하는 NaN 문제 방지
        return torch.sigmoid(x)
