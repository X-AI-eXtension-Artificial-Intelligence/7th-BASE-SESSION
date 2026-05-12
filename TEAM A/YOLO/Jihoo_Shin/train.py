"""
train.py  ─  YOLOv1 단일 이미지 과적합(overfit) 학습 예시
──────────────────────────────────────────────────────────────
사용법:
  python train.py                           # 합성 이미지로 학습
  python train.py --image data/my_photo.jpg # 실제 이미지로 학습

목적:
  • 이미지 1장에 대해 loss 가 수렴하는지 확인
  • 전체 파이프라인(모델 → loss → 역전파 → 저장) 검증
"""

import argparse
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model   import YOLOv1
from loss    import YOLOv1Loss
from dataset import create_demo_dataset

# ─── 하이퍼파라미터 ────────────────────────────────────────────
S       = 7
B       = 2
C       = 20
LR      = 1e-4
EPOCHS  = 50
BATCH   = 4
REPEAT  = 200      # 단일 이미지 반복 횟수 (에포크당 샘플 수)
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = 'weights'


def parse_args():
    p = argparse.ArgumentParser(description='YOLOv1 단일 이미지 학습')
    p.add_argument('--image',  type=str, default=None,
                   help='학습에 사용할 이미지 경로 (없으면 합성 이미지 사용)')
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--lr',     type=float, default=LR)
    p.add_argument('--batch',  type=int, default=BATCH)
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    log_dict   = {'box': 0, 'obj': 0, 'noobj': 0, 'cls': 0}
    t0 = time.time()

    for imgs, targets in loader:
        imgs    = imgs.to(device)
        targets = targets.to(device)

        preds = model(imgs)                            # (N, S, S, C+B*5)
        loss, parts = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        for k in log_dict:
            log_dict[k] += parts[k]

    n = len(loader)
    elapsed = time.time() - t0
    print(f"[Epoch {epoch:03d}] "
          f"loss={total_loss/n:.4f}  "
          f"box={log_dict['box']/n:.3f}  "
          f"obj={log_dict['obj']/n:.3f}  "
          f"noobj={log_dict['noobj']/n:.3f}  "
          f"cls={log_dict['cls']/n:.3f}  "
          f"({elapsed:.1f}s)")
    return total_loss / n


def main():
    args = parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── 데이터셋 ──────────────────────────────────────────────
    dataset = create_demo_dataset(image_path=args.image, repeat=REPEAT)
    loader  = DataLoader(dataset, batch_size=args.batch,
                         shuffle=True, num_workers=0)

    # ── 모델 ──────────────────────────────────────────────────
    model = YOLOv1(S=S, B=B, C=C, mode='detection').to(DEVICE)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # ── 손실 / 옵티마이저 ─────────────────────────────────────
    criterion = YOLOv1Loss(S=S, B=B, C=C)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')

    # ── 학습 루프 ─────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, criterion, optimizer, DEVICE, epoch)
        scheduler.step()

        # 체크포인트 저장
        if loss < best_loss:
            best_loss = loss
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'loss':  best_loss,
                'S': S, 'B': B, 'C': C,
            }
            torch.save(ckpt, os.path.join(SAVE_DIR, 'best.pt'))
            print(f"  ✓ best 모델 저장 (loss={best_loss:.4f})")

    # 마지막 체크포인트
    torch.save({
        'epoch': args.epochs,
        'model': model.state_dict(),
        'S': S, 'B': B, 'C': C,
    }, os.path.join(SAVE_DIR, 'last.pt'))
    print("\n학습 완료! weights/best.pt 에 최고 모델이 저장되었습니다.")


if __name__ == '__main__':
    main()
