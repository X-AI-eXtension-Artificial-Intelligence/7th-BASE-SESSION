"""
config.py
모든 하이퍼파라미터를 한 곳에서 관리한다.
"""

import torch

CFG = {
    # ── 데이터 ──────────────────────────────
    "img_size"      : 32,
    "in_channels"   : 3,
    "num_classes"   : 10,

    # ── 패치 ────────────────────────────────
    "patch_size"    : 4,     # (32/4)² = 64 patches

    # ── Transformer ─────────────────────────
    "d_model"       : 128,
    "depth"         : 6,
    "n_heads"       : 8,
    "mlp_ratio"     : 4,
    "dropout"       : 0.1,

    # ── 학습 ────────────────────────────────
    "epochs"        : 15,
    "batch_size"    : 128,
    "lr"            : 1e-3,
    "weight_decay"  : 0.05,
    "warmup_epochs" : 5,
    "label_smoothing": 0.1,
    "grad_clip"     : 1.0,

    # ── 기타 ────────────────────────────────
    "num_workers"   : 2,
    "save_path"     : "vit_cifar10_best.pth",
    "device"        : (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ),
}

# 파생 값 (config 로드 후 자동 계산)
CFG["n_patches"] = (CFG["img_size"] // CFG["patch_size"]) ** 2   # 64
CFG["seq_len"]   = CFG["n_patches"] + 1                           # 65 (+CLS)