"""
U-Net Training Script
논문: Ronneberger et al., 2015

실행 예시:
  python train.py --mode train --num_epoch 50 --lr 1e-4
  python train.py --mode test  --ckpt_path ./checkpoints/best.pth
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import UNet
from dataset import SegmentationDataset, Normalize, RandomFlip, ElasticDeformation, ToTensor


# ─────────────────────────────────────────────
# CLI 인자
# ─────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description='U-Net Training (Ronneberger et al., 2015)')
    p.add_argument('--mode',         type=str,   default='train', choices=['train', 'test'])
    p.add_argument('--data_dir',     type=str,   default='./datasets')
    p.add_argument('--ckpt_dir',     type=str,   default='./checkpoints')
    p.add_argument('--result_dir',   type=str,   default='./results')
    p.add_argument('--ckpt_path',    type=str,   default=None,
                   help='test 모드 또는 이어서 학습 시 로드할 체크포인트 경로')
    p.add_argument('--lr',           type=float, default=1e-4,
                   help='논문: SGD momentum 0.99 사용. 본 구현은 Adam 기본값')
    p.add_argument('--batch_size',   type=int,   default=2)
    p.add_argument('--num_epoch',    type=int,   default=50)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--use_bn',       action='store_true', default=True,
                   help='BatchNorm 사용 여부 (논문 원본 미사용, 실용적으로 권장)')
    return p.parse_args()


# ─────────────────────────────────────────────
# 체크포인트 저장 / 로드
# ─────────────────────────────────────────────
def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  [저장] {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    start_epoch = ckpt.get('epoch', 0)
    best_val    = ckpt.get('best_val_loss', float('inf'))
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f"  [로드] {path}  (epoch {start_epoch}, val_loss {best_val:.4f})")
    return start_epoch, best_val


# ─────────────────────────────────────────────
# 결과 저장 헬퍼
# ─────────────────────────────────────────────
def save_results(result_dir, batch_idx, image, label, output):
    png_dir = os.path.join(result_dir, 'png')
    npy_dir = os.path.join(result_dir, 'numpy')
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    image_np  = image.cpu().numpy()[0, 0]
    label_np  = label.cpu().numpy()[0, 0]
    output_np = (torch.sigmoid(output).cpu().numpy()[0, 0] > 0.5).astype(np.float32)

    for name, arr in [('image', image_np), ('label', label_np), ('output', output_np)]:
        np.save(os.path.join(npy_dir, f'{name}_{batch_idx:03d}.npy'), arr)
        plt.imsave(os.path.join(png_dir, f'{name}_{batch_idx:03d}.png'), arr, cmap='gray')


# ─────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────
def train(args, model, device):
    # transforms
    train_tf = transforms.Compose([
        ElasticDeformation(alpha=34.0, sigma=4.0, p=0.5),  # 논문 핵심 augmentation
        RandomFlip(),
        Normalize(mean=0.5, std=0.5),
        ToTensor(),
    ])
    val_tf = transforms.Compose([
        Normalize(mean=0.5, std=0.5),
        ToTensor(),
    ])

    train_ds = SegmentationDataset(os.path.join(args.data_dir, 'train'), transform=train_tf)
    val_ds   = SegmentationDataset(os.path.join(args.data_dir, 'val'),   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)}장  |  Val: {len(val_ds)}장")

    # 논문: SGD + momentum 0.99 / 본 구현: Adam (더 안정적)
    # 논문 재현 시 아래 주석 해제:
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 픽셀별 이진 분류 → BCEWithLogitsLoss (내부적으로 sigmoid 포함)
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_val_loss = float('inf')

    # 이어서 학습
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        start_epoch, best_val_loss = load_checkpoint(
            args.ckpt_path, model, optimizer, device
        )

    train_losses, val_losses = [], []

    for epoch in range(start_epoch, args.num_epoch):
        # ── Train ──
        model.train()
        epoch_loss = []
        for batch in train_loader:
            image = batch['image'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            output = model(image)
            loss   = criterion(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        avg_train = np.mean(epoch_loss)
        train_losses.append(avg_train)

        # ── Validation ──
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(device)
                label = batch['label'].to(device)
                output = model(image)
                val_loss.append(criterion(output, label).item())

        avg_val = np.mean(val_loss)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        print(f"Epoch {epoch+1:3d}/{args.num_epoch} | "
              f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # best 체크포인트 저장
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(args.ckpt_dir, 'best.pth'))

        # 매 10 epoch 마다 정기 저장
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(args.ckpt_dir, f'epoch_{epoch+1:03d}.pth'))

    # 학습 곡선 저장
    _plot_loss(train_losses, val_losses, args.result_dir)
    print("\n학습 완료")


# ─────────────────────────────────────────────
# 테스트 루프
# ─────────────────────────────────────────────
def test(args, model, device):
    if not args.ckpt_path:
        # ckpt_path 미지정 시 best.pth 자동 탐색
        args.ckpt_path = os.path.join(args.ckpt_dir, 'best.pth')

    load_checkpoint(args.ckpt_path, model, device=device)

    test_tf = transforms.Compose([
        Normalize(mean=0.5, std=0.5),
        ToTensor(),
    ])
    test_ds     = SegmentationDataset(os.path.join(args.data_dir, 'test'), transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    test_losses = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image  = batch['image'].to(device)
            label  = batch['label'].to(device)
            output = model(image)

            loss = criterion(output, label)
            test_losses.append(loss.item())

            save_results(args.result_dir, idx, image, label, output)

    print(f"Test Loss (평균): {np.mean(test_losses):.4f}")
    print(f"결과 저장: {args.result_dir}")


# ─────────────────────────────────────────────
# 학습 곡선 시각화
# ─────────────────────────────────────────────
def _plot_loss(train_losses, val_losses, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('U-Net Training Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'loss_curve.png'), dpi=150)
    plt.close()
    print(f"  [저장] {result_dir}/loss_curve.png")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    args = get_args()

    # 재현성
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = UNet(in_channels=1, out_channels=1, features=64, use_bn=args.use_bn).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {total_params:,}")

    os.makedirs(args.result_dir, exist_ok=True)

    if args.mode == 'train':
        train(args, model, device)
    else:
        test(args, model, device)
