"""
main.py
실행 진입점.

사용법:
    python main.py
"""

import time
import torch
import torch.nn as nn

from config import CFG
from data   import get_dataloaders
from vit    import build_vit
from train  import train_one_epoch, evaluate
from utils  import get_scheduler, print_model_info, show_attention


def main():
    device = CFG["device"]

    # ── 1. 데이터 ────────────────────────────
    print("\n[1] 데이터셋 준비 중...")
    train_loader, test_loader = get_dataloaders(CFG)

    # ── 2. 모델 ──────────────────────────────
    print("\n[2] 모델 초기화...")
    model = build_vit(CFG).to(device)
    print_model_info(model, CFG)

    # ── 3. 옵티마이저 & 스케줄러 ─────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
        betas        = (0.9, 0.999),
    )
    scheduler = get_scheduler(
        optimizer,
        warmup_epochs   = CFG["warmup_epochs"],
        total_epochs    = CFG["epochs"],
        steps_per_epoch = len(train_loader),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

    # ── 4. 학습 루프 ─────────────────────────
    print("\n[3] 학습 시작\n")
    header = (f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>10} | "
              f"{'Val Loss':>9} | {'Val Acc':>8} | {'LR':>9} | {'Time':>7}")
    divider = "-" * len(header)
    print(header)
    print(divider)

    best_acc = 0.0

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, CFG
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, CFG)

        elapsed    = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.2f}% | "
            f"{val_loss:>9.4f} | {val_acc:>7.2f}% | "
            f"{current_lr:>9.6f} | {elapsed:>6.1f}s"
        )

        # 최고 성능 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CFG["save_path"])

    print(divider)
    print(f"\n학습 완료.  Best Val Accuracy: {best_acc:.2f}%")
    print(f"모델 저장 경로: {CFG['save_path']}")

    # ── 5. Attention Rollout 시각화 ───────────
    print("\n[4] Attention Rollout 시각화...")
    show_attention(model, test_loader, CFG)


if __name__ == "__main__":
    main()
