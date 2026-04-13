"""
U-Net 학습 스크립트

논문 기반 학습 설정:
- Optimizer: SGD with momentum=0.99 (논문 Section 3)
- Loss: Weighted Cross Entropy (논문 Eq.1)
- 초기 lr: 0.01, StepLR decay
- Data Augmentation: 수평 뒤집기, 랜덤 회전, 색상 변환 (논문 Section 3.1)
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로 저장
import matplotlib.pyplot as plt

from unet_model import UNet
from dataset import get_dataloaders


# ─── 평가 메트릭 ──────────────────────────────────────────────
def dice_coefficient(pred, target, smooth=1e-6):
    """Dice Coefficient = 2|A∩B| / (|A|+|B|)"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(pred, target, smooth=1e-6):
    """IoU (Jaccard Index) = |A∩B| / |A∪B|"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def pixel_accuracy(pred, target):
    """Pixel-wise Accuracy"""
    pred = (pred > 0.5).long()
    correct = (pred == target).float().mean()
    return correct.item()


# ─── 학습 루프 ────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)  # [B, 2, H, W]

        # Cross entropy loss
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        # 확률로 변환 (foreground channel)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        total_loss += loss.item()
        total_dice += dice_coefficient(probs, masks.float())
        total_iou += iou_score(probs, masks.float())

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_acc = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, masks)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        total_loss += loss.item()
        total_dice += dice_coefficient(probs, masks.float())
        total_iou += iou_score(probs, masks.float())
        total_acc += pixel_accuracy(probs, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n, total_acc / n


# ─── 결과 시각화 ──────────────────────────────────────────────
def plot_history(history, save_path='results/training_history.png'):
    os.makedirs('results', exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history['train_dice'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_dice'], 'r-', label='Val')
    axes[1].set_title('Dice Coefficient')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, history['train_iou'], 'b-', label='Train')
    axes[2].plot(epochs, history['val_iou'], 'r-', label='Val')
    axes[2].set_title('IoU Score')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle('U-Net Training History (Oxford Pet Dataset)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"학습 곡선 저장: {save_path}")


@torch.no_grad()
def visualize_predictions(model, loader, device, save_path='results/predictions.png', n_samples=4):
    """예측 결과 시각화 (원본 이미지 / Ground Truth / 예측)"""
    os.makedirs('results', exist_ok=True)
    model.eval()

    imgs, masks = next(iter(loader))
    imgs = imgs.to(device)
    outputs = model(imgs)
    probs = torch.softmax(outputs, dim=1)[:, 1].cpu()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 4))
    titles = ['Input Image', 'Ground Truth', 'U-Net Prediction']

    for i in range(min(n_samples, len(imgs))):
        img_vis = (imgs[i].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        gt = masks[i].cpu().numpy()
        pred = (probs[i] > 0.5).numpy().astype(float)

        for j, (data, title) in enumerate(zip([img_vis, gt, pred], titles)):
            ax = axes[i][j]
            if j == 0:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap='gray', vmin=0, vmax=1)
            ax.set_title(title if i == 0 else '')
            ax.axis('off')

    plt.suptitle('U-Net Segmentation Results', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"예측 결과 저장: {save_path}")


# ─── 메인 학습 ────────────────────────────────────────────────
def train(
    num_epochs=20,
    batch_size=8,
    lr=0.01,
    image_size=256,
    num_classes=2,
    save_dir='checkpoints',
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"디바이스: {device}")

    # 데이터 (max_train=800으로 제한하여 실험 속도 향상)
    train_loader, val_loader, test_loader = get_dataloaders(
        image_size=image_size, batch_size=batch_size,
        max_train=800, max_test=400,
    )

    # 모델
    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total_params:,}")

    # 손실함수 - 클래스 가중치 (논문: weighted loss)
    # pet(foreground)가 background보다 적으므로 가중치 부여
    class_weights = torch.tensor([0.4, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 옵티마이저 - 논문: SGD with momentum=0.99
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [], 'val_iou': [],
    }

    best_dice = 0
    print(f"\n{'='*60}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8} | {'Train Dice':>10} | {'Val Dice':>8} | {'Val IoU':>7} | {'Time':>6}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_dice)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)

        elapsed = time.time() - t0
        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>8.4f} | {train_dice:>10.4f} | {val_dice:>8.4f} | {val_iou:>7.4f} | {elapsed:>5.1f}s")

        # Best model 저장
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  → Best model saved (Dice: {best_dice:.4f})")

    print(f"{'='*60}")
    print(f"\n최고 Val Dice: {best_dice:.4f}")

    # 최적 모델 로드 후 테스트셋 평가
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_dice, test_iou, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*40}")
    print(f"[Test Set 최종 성능]")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Dice:     {test_dice:.4f}")
    print(f"  IoU:      {test_iou:.4f}")
    print(f"  Pixel Acc:{test_acc:.4f}")
    print(f"{'='*40}")

    # 시각화
    plot_history(history)
    visualize_predictions(model, test_loader, device)

    return model, history


if __name__ == '__main__':
    # image_size=128로 줄여 MPS에서 빠른 학습
    train(num_epochs=20, batch_size=16, lr=0.01, image_size=128)
