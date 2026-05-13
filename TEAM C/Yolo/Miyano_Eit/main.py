import os
import torch
import torch.optim as optim
from model import get_model
from loss import YOLOLoss
from data_loader import get_synthetic_dataloader

os.makedirs("etc", exist_ok=True)

S = 7
B = 2
C = 20
EPOCHS = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(S=S, B=B, C=C).to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-5,
                      momentum=0.9, weight_decay=5e-4)
criterion = YOLOLoss(S=S, B=B, C=C)

# 합성 데이터 1000장 생성
train_loader = get_synthetic_dataloader(
    num_samples=1000, S=S, B=B, C=C, batch_size=16, img_size=112
)

print(f"합성 데이터 수: 1000장")
print(f"배치 수: {len(train_loader)}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    valid_batches = 0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, targets)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        valid_batches += 1

    if valid_batches > 0:
        avg_loss = epoch_loss / valid_batches
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "etc/yolo.pt")
print("모델 저장 완료 → etc/yolo.pt")