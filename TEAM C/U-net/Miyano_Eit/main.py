import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import UNet
from train import train_epoch, evaluate
from data_loader import get_dataloader

os.makedirs("etc", exist_ok=True)

# --- 설정 ---
IN_CHANNELS  = 3
NUM_CLASSES  = 2        # 배경 + 병변
IMG_SIZE     = 128
BATCH_SIZE   = 8
EPOCHS       = 30
LR           = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 모델 ---
model = UNet(
    in_channels=IN_CHANNELS,
    num_classes=NUM_CLASSES,
    features=[64, 128, 256, 512]    # 논문 기본 설정
).to(device)

print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# --- 데이터 ---
train_loader = get_dataloader(
    num_samples=500, img_size=IMG_SIZE,
    num_classes=NUM_CLASSES, in_channels=IN_CHANNELS,
    batch_size=BATCH_SIZE
)

# --- 학습 설정 ---
# CrossEntropyLoss: 픽셀별 분류 문제
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)

# --- 학습 ---
print("\n학습 시작...")
best_loss = float('inf')
patience = 3        # 3 epoch 연속 개선 없으면 종료
no_improve = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, criterion, device
    )
    scheduler.step(train_loss)

    print(f"Epoch {epoch:02d} | "
          f"Loss: {train_loss:.4f} | "
          f"Pixel Acc: {train_acc:.1f}%")

    if train_loss < best_loss:
        best_loss = train_loss
        no_improve = 0
        torch.save(model.state_dict(), "etc/unet_best.pt")
    else:
        no_improve += 1

    # Pixel Acc 100% + loss 충분히 낮으면 종료
    if train_acc >= 100.0 and train_loss < 0.01:
        print(f"\nEarly stopping: Pixel Acc 100%, Loss {train_loss:.4f}")
        break

    if no_improve >= patience:
        print(f"\nEarly stopping: {patience} epoch 동안 개선 없음")
        break


torch.save(model.state_dict(), "etc/unet_last.pt")
print(f"\n학습 완료")
print(f"최고 Loss: {best_loss:.4f}")
print(f"모델 저장 → etc/unet_best.pt")