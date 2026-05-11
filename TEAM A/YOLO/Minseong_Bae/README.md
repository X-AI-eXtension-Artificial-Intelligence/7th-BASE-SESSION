# YOLOv1 구현

논문 **"You Only Look Once: Unified, Real-Time Object Detection"** (Redmon et al., CVPR 2016)을 기반으로 PyTorch로 구현한 YOLOv1입니다.

---

## 파일 구조

```
4주차/
├── yolo_v1.py   # 전체 구현 (모델, 손실함수, 학습 루프)
├── YOLO.pdf     # 원본 논문
└── README.md
```

---

## YOLO의 핵심 아이디어

기존 객체 탐지는 분류기(classifier)를 재활용하는 방식이었습니다. YOLO는 이를 **단일 회귀 문제**로 재정의합니다.

> 이미지 전체를 한 번에 보고, bounding box 좌표와 클래스 확률을 동시에 예측한다.

1. 입력 이미지를 **S×S 그리드**로 분할한다.
2. 각 그리드 셀이 **B개의 bounding box**와 **C개의 클래스 확률**을 예측한다.
3. 하나의 네트워크가 이 모든 것을 **한 번의 forward pass**로 출력한다.

---

## 아키텍처 (`YOLOv1`, `ConvBlock`)

논문 Figure 3의 구조를 그대로 따릅니다.

```
입력: 448 × 448 × 3
│
├─ [Layer 1]   Conv 7×7, 64ch, stride=2  →  224×224
│              MaxPool 2×2, stride=2      →  112×112
│
├─ [Layer 2]   Conv 3×3, 192ch           →  112×112
│              MaxPool 2×2, stride=2      →   56×56
│
├─ [Layer 3-6] Conv 1×1×128, 3×3×256, 1×1×256, 3×3×512
│              MaxPool 2×2, stride=2      →   28×28
│
├─ [Layer 7-16] { Conv 1×1×256, Conv 3×3×512 } × 4
│               Conv 1×1×512, Conv 3×3×1024
│               MaxPool 2×2, stride=2     →   14×14
│
├─ [Layer 17-22] { Conv 1×1×512, Conv 3×3×1024 } × 2
│                Conv 3×3×1024
│                Conv 3×3×1024, stride=2  →    7×7
│
├─ [Layer 23-24] Conv 3×3×1024 × 2       →    7×7
│
├─ [FC 1]  Flatten → Linear(7·7·1024, 4096) → Dropout(0.5) → Leaky ReLU
└─ [FC 2]  Linear(4096, 7·7·30)

출력: 7 × 7 × 30
```

**ConvBlock** = Conv2d → Leaky ReLU(slope=0.1)
(YOLOv1 원본에는 Batch Normalization 없음 — BN은 v2에서 추가됨)

### 출력 텐서 해석

출력 shape `(N, 7, 7, 30)` 에서 마지막 차원 30의 구성:

```
[ x, y, w, h, conf ]  ← Box 0 (5개)
[ x, y, w, h, conf ]  ← Box 1 (5개)
[ cls_0, cls_1, ..., cls_19 ]  ← 클래스 확률 (20개)
```

| 값 | 의미 |
|---|---|
| `x, y` | 셀 기준 중심 좌표 (0~1 범위) |
| `w, h` | 이미지 전체 기준 크기 (0~1 범위) |
| `conf` | 물체 존재 여부 × 예측 박스의 GT와의 IoU |
| `cls_i` | 물체가 있을 때 클래스 i일 조건부 확률 |

---

## 손실 함수 (`YOLOv1Loss`)

논문 Equation 3을 그대로 구현합니다.

$$
\mathcal{L} =
\lambda_\text{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbf{1}_{ij}^\text{obj}
\left[(x_i - \hat x_i)^2 + (y_i - \hat y_i)^2 \right]
$$

$$
+ \lambda_\text{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbf{1}_{ij}^\text{obj}
\left[(\sqrt{w_i} - \sqrt{\hat w_i})^2 + (\sqrt{h_i} - \sqrt{\hat h_i})^2 \right]
$$

$$
+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbf{1}_{ij}^\text{obj} (C_i - \hat C_i)^2
$$

$$
+ \lambda_\text{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbf{1}_{ij}^\text{noobj} (C_i - \hat C_i)^2
$$

$$
+ \sum_{i=0}^{S^2} \mathbf{1}_{i}^\text{obj} \sum_{c \in \text{classes}} (p_i(c) - \hat p_i(c))^2
$$

손실은 **5개 항목**으로 구성됩니다.

| 항목 | 대상 | 가중치 | 설명 |
|---|---|---|---|
| 좌표 (x, y) | 물체 있는 셀의 responsible box | λ_coord = **5** | 중심 좌표 오차 |
| 크기 (w, h) | 물체 있는 셀의 responsible box | λ_coord = **5** | √를 씌워 작은 박스에 더 민감하게 |
| Confidence (obj) | 물체 있는 셀의 responsible box | 1 | 예측 신뢰도 오차 |
| Confidence (noobj) | 물체 없는 셀의 모든 box | λ_noobj = **0.5** | 배경 셀의 과도한 학습 억제 |
| 클래스 확률 | 물체 있는 셀 | 1 | 클래스 분류 오차 |

### Responsible Box 선택

각 셀에서 B개의 box 예측자 중 **GT와 IoU가 가장 높은 box 하나**만 coordinate/confidence loss에 기여합니다. 이 메커니즘이 각 predictor를 서로 다른 크기·비율의 박스에 특화되도록 유도합니다.

### √(w, h) 처리

큰 박스와 작은 박스의 오차를 균등하게 다루기 위해 w, h에 제곱근을 적용합니다. 예측값이 음수일 수 있으므로 다음과 같이 처리합니다.

```python
# 부호를 보존하면서 안전하게 제곱근 적용
pred_w = pred_boxes[..., 2].abs().sqrt() * pred_boxes[..., 2].sign()
```

---

## 학습 설정 (`train`, `build_lr_scheduler`)

논문 §2.2의 학습 절차를 따릅니다.

### Optimizer

```python
optim.SGD(lr=1e-2, momentum=0.9, weight_decay=5e-4)
```

### Learning Rate Schedule

| 구간 | Learning Rate |
|---|---|
| Warm-up (1 epoch) | 1e-3 → 1e-2 선형 증가 |
| 75 epoch | 1e-2 |
| 30 epoch | 1e-3 |
| 30 epoch | 1e-4 |

초기에 높은 lr로 시작하면 gradient 발산이 생기기 때문에 warm-up을 사용합니다.

### 기타 설정

| 항목 | 값 |
|---|---|
| Batch size | 64 |
| 총 epoch | ~135 |
| Dropout | 0.5 (첫 번째 FC 이후) |
| 데이터 증강 | 랜덤 스케일/이동 (원본 대비 ±20%), HSV 채도·밝기 조절 (±1.5배) |

---

## 데이터셋 연동 (`VOCDataset`)

`VOCDataset`은 실제 데이터셋 연결을 위한 껍데기(stub)입니다. `__len__`과 `__getitem__`을 구현한 뒤 `build_target`으로 GT 텐서를 만들면 됩니다.

### `build_target` 동작 방식

```
boxes     = [[0.5, 0.4, 0.3, 0.2], ...]   # x_c, y_c, w, h (이미지 기준 0~1)
class_ids = [14, ...]                       # 클래스 인덱스

→ target[row][col] = [x_cell, y_cell, w, h, 1.0,   # box 0
                       x_cell, y_cell, w, h, 1.0,   # box 1 (동일)
                       0, 0, ..., 1, ..., 0]         # one-hot class
```

- `x_cell = x_c × S - col` : 셀 내부 상대 좌표 (0~1)
- w, h는 이미지 전체 기준 그대로 유지
- 한 셀에 여러 물체가 있으면 첫 번째 물체만 기록 (YOLOv1 한계)

---

## 실행 방법

### 동작 확인 (shape & loss 검사)

```bash
python yolo_v1.py
# Device: cpu
# Forward pass OK — output shape: (2, 7, 7, 30)
# Loss check OK  — loss: 0.0039
```

### 실제 학습 시작

`yolo_v1.py` 하단 주석을 해제하고 데이터 경로를 지정합니다.

```python
# VOCDataset의 __len__, __getitem__ 구현 후:

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])
dataset = VOCDataset("data/images", "data/labels", transform=transform)
loader  = DataLoader(dataset, batch_size=64, shuffle=True,
                     num_workers=4, pin_memory=True)
train(model, loader, device)
```

---

## 의존성

```bash
pip install torch torchvision
```

---

## 참고

- 논문: Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016.
- 이 구현은 YOLOv1 원본 논문만을 기반으로 합니다. Batch Normalization, anchor box 등 이후 버전(v2, v3)의 개선 사항은 포함하지 않습니다.
