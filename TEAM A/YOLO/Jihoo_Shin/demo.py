"""
demo.py  ─  단일 스크립트로 전체 파이프라인을 한 번에 실행
─────────────────────────────────────────────────────────────
pip install torch torchvision opencv-python 만 설치되어 있으면
추가 데이터셋/가중치 없이 바로 실행 가능합니다.

실행:
  python demo.py
"""

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2

# ─── 설정 ────────────────────────────────────────────────────
S        = 7
B        = 2
C        = 20
IMG_SIZE = 448
EPOCHS   = 30
BATCH    = 4
LR       = 2e-4
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs('outputs', exist_ok=True)
os.makedirs('weights', exist_ok=True)

# ─── 로컬 모듈 임포트 ─────────────────────────────────────────
from model   import YOLOv1
from loss    import YOLOv1Loss
from dataset import DemoSyntheticDataset
from utils   import decode_predictions, nms, draw_boxes, encode_target

print("=" * 55)
print("  YOLOv1 Demo  (단일 이미지 학습 + 추론)")
print(f"  Device: {DEVICE}")
print("=" * 55)

# ─────────────────────────────────────────────────────────────
# STEP 1. 합성 이미지와 GT 박스 정의
# ─────────────────────────────────────────────────────────────
GT_BOXES = [
    (112, 56,  336, 392, 14),   # person  (class 14)
    (22,  250, 180, 420, 11),   # dog     (class 11)
]

# 시각화용 실제 이미지도 생성 (draw_boxes에서 사용)
demo_img = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
for (x1, y1, x2, y2, cls_id) in GT_BOXES:
    color = np.array([(cls_id * 37 % 255) / 255,
                       (cls_id * 73 % 255) / 255,
                       (cls_id * 113 % 255) / 255], dtype=np.float32)
    demo_img[y1:y2, x1:x2] = color

# GT 박스를 이미지에 그려서 저장
demo_img_uint8 = (demo_img * 255).astype(np.uint8)
demo_img_bgr   = cv2.cvtColor(demo_img_uint8, cv2.COLOR_RGB2BGR)
for (x1, y1, x2, y2, cls_id) in GT_BOXES:
    cv2.rectangle(demo_img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(demo_img_bgr, f"GT cls={cls_id}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imwrite('outputs/demo_input_with_gt.jpg', demo_img_bgr)
print("\n[Step 1] 합성 이미지 생성 → outputs/demo_input_with_gt.jpg")

# ─────────────────────────────────────────────────────────────
# STEP 2. 데이터셋 / 모델 / 손실 / 옵티마이저
# ─────────────────────────────────────────────────────────────
dataset   = DemoSyntheticDataset(GT_BOXES, S=S, C=C, repeat=200)
loader    = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)
model     = YOLOv1(S=S, B=B, C=C, mode='detection').to(DEVICE)
criterion = YOLOv1Loss(S=S, B=B, C=C)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"\n[Step 2] 모델 파라미터: {sum(p.numel() for p in model.parameters()):,}개")

# ─────────────────────────────────────────────────────────────
# STEP 3. 학습
# ─────────────────────────────────────────────────────────────
print(f"\n[Step 3] 학습 시작 ({EPOCHS} 에포크)")
best_loss = float('inf')
loss_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for imgs, targets in loader:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        preds = model(imgs)
        loss, _ = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    avg = epoch_loss / len(loader)
    loss_history.append(avg)

    if avg < best_loss:
        best_loss = avg
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'S': S, 'B': B, 'C': C}, 'weights/best.pt')

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch [{epoch:03d}/{EPOCHS}]  loss={avg:.4f}  best={best_loss:.4f}")

print(f"\n  최종 loss: {loss_history[-1]:.4f}  (초기 loss: {loss_history[0]:.4f})")
print("  가중치 저장 → weights/best.pt")

# ─────────────────────────────────────────────────────────────
# STEP 4. 추론
# ─────────────────────────────────────────────────────────────
print("\n[Step 4] 추론")
model.eval()

# 합성 이미지를 텐서로 변환
img_tensor = torch.from_numpy(demo_img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(img_tensor)   # (1, S, S, C+B*5)

# 디코딩 + NMS
raw_boxes = decode_predictions(output, S=S, B=B, C=C, conf_thresh=0.2)[0]
final_boxes = nms(raw_boxes, iou_thresh=0.5)

print(f"  감지된 객체: {len(final_boxes)}개")
from utils import VOC_CLASSES
for (x1, y1, x2, y2, score, cls_id) in final_boxes:
    print(f"    [{VOC_CLASSES[cls_id]}] score={score:.3f}  "
          f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

# 결과 시각화
result_img = draw_boxes(demo_img_bgr.copy(), final_boxes)
cv2.imwrite('outputs/demo_result.jpg', result_img)
print("  결과 저장 → outputs/demo_result.jpg")

# ─────────────────────────────────────────────────────────────
# STEP 5. loss 곡선 저장 (matplotlib 없으면 텍스트로 대체)
# ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('YOLOv1 Loss (Single Image Overfit)')
    plt.legend(); plt.tight_layout()
    plt.savefig('outputs/loss_curve.png', dpi=120)
    plt.close()
    print("\n[Step 5] Loss 곡선 저장 → outputs/loss_curve.png")
except ImportError:
    with open('outputs/loss_history.txt', 'w') as f:
        for i, v in enumerate(loss_history, 1):
            f.write(f"Epoch {i}: {v:.6f}\n")
    print("\n[Step 5] Loss 기록 저장 → outputs/loss_history.txt")

print("\n✅ 전체 파이프라인 완료!")
print("   outputs/demo_input_with_gt.jpg  ← 입력 이미지 + GT 박스")
print("   outputs/demo_result.jpg         ← 모델 예측 결과")
print("   weights/best.pt                 ← 학습된 가중치")
