# U-Net 구현 및 성능 도출

> Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015 (arXiv:1505.04597)

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [파일 구조](#2-파일-구조)
3. [아키텍처 설명 (unet_model.py)](#3-아키텍처-설명-unet_modelpy)
4. [데이터셋 (dataset.py)](#4-데이터셋-datasetpy)
5. [학습 로직 (train.py)](#5-학습-로직-trainpy)
6. [전체 데이터 흐름](#6-전체-데이터-흐름)
7. [실험 결과](#7-실험-결과)
8. [실행 방법](#8-실행-방법)

---

## 1. 프로젝트 개요

U-Net 논문을 기반으로 PyTorch로 직접 구현하고, **Oxford-IIIT Pet Dataset**에서 반려동물 이진 segmentation(pet vs background) 성능을 도출한 프로젝트입니다.

### 논문의 핵심 아이디어

- **Contracting Path (Encoder)**: 3×3 conv × 2 + ReLU + 2×2 max pooling으로 문맥(context) 정보를 압축
- **Expansive Path (Decoder)**: 2×2 up-convolution으로 해상도를 복원하며 정밀한 위치 정보 생성
- **Skip Connection**: Encoder의 feature map을 Decoder에 직접 연결(concat)하여 위치 정보 손실 방지
- **U자형 구조**: 두 경로가 대칭을 이루어 U 모양을 형성

---

## 2. 파일 구조

```
1주차/
├── unet_model.py          # U-Net 아키텍처 정의
├── dataset.py             # Oxford Pet Dataset 로더
├── train.py               # 학습 + 평가 스크립트
├── checkpoints/
│   └── best_model.pth     # 최적 모델 가중치 (Val Dice 기준)
├── results/
│   ├── training_history.png   # 학습 곡선 (Loss / Dice / IoU)
│   └── predictions.png        # 예측 결과 시각화
└── data/
    └── oxford-iiit-pet/       # 자동 다운로드되는 데이터셋
```

---

## 3. 아키텍처 설명 (`unet_model.py`)

### 3-1. DoubleConv 블록

U-Net의 모든 단계에서 반복 사용되는 기본 단위입니다.

```
입력
 → Conv2d(3×3, padding=1)
 → BatchNorm2d
 → ReLU
 → Conv2d(3×3, padding=1)
 → BatchNorm2d
 → ReLU
출력 (입출력 공간 크기 동일)
```

- `padding=1`: 논문 원본은 unpadded이지만, 입출력 크기를 동일하게 유지하기 위해 padding 적용
- `bias=False`: BatchNorm이 bias 역할을 대신하므로 생략
- `inplace=True`: ReLU를 제자리 연산으로 수행하여 메모리 절약

### 3-2. 전체 U-Net 구조

```
입력 이미지 [B, 3, H, W]
│
├─ Encoder (Contracting Path)
│   ├─ enc1: DoubleConv(3→64)      → e1 [B, 64, H, W]
│   │         MaxPool2d(2×2)
│   ├─ enc2: DoubleConv(64→128)    → e2 [B, 128, H/2, W/2]
│   │         MaxPool2d(2×2)
│   ├─ enc3: DoubleConv(128→256)   → e3 [B, 256, H/4, W/4]
│   │         MaxPool2d(2×2)
│   └─ enc4: DoubleConv(256→512)   → e4 [B, 512, H/8, W/8]
│             MaxPool2d(2×2)
│
├─ Bottleneck
│   └─ DoubleConv(512→1024)        → b  [B, 1024, H/16, W/16]
│
└─ Decoder (Expansive Path)
    ├─ up4: ConvTranspose2d(1024→512)
    │   cat([up4(b), e4]) → dec4: DoubleConv(1024→512) → d4 [B, 512, H/8, W/8]
    ├─ up3: ConvTranspose2d(512→256)
    │   cat([up3(d4), e3]) → dec3: DoubleConv(512→256) → d3 [B, 256, H/4, W/4]
    ├─ up2: ConvTranspose2d(256→128)
    │   cat([up2(d3), e2]) → dec2: DoubleConv(256→128) → d2 [B, 128, H/2, W/2]
    ├─ up1: ConvTranspose2d(128→64)
    │   cat([up1(d2), e1]) → dec1: DoubleConv(128→64)  → d1 [B, 64, H, W]
    │
    └─ final: Conv2d(64→num_classes, 1×1)
                                    → [B, num_classes, H, W]
```

**Skip Connection의 역할**

Decoder의 각 단계에서 `torch.cat([up(x), encoder_feature], dim=1)`으로 채널을 이어붙입니다.
- Encoder: 맥락 정보(무엇이 있는가)
- Decoder: 위치 정보(어디에 있는가)
- 두 정보를 합쳐 정밀한 segmentation map 생성

DoubleConv 입력 채널이 두 배인 이유도 이 때문입니다 (예: upsample 512 + skip 512 = 1024).

### 3-3. 가중치 초기화 (논문 Section 3)

논문에서 명시한 **He initialization** 적용:

```
std = √(2 / N)    (N = 입력 연결 수)
```

ReLU 활성화 함수를 사용할 때 gradient 소실 없이 학습이 안정적으로 시작되도록 합니다.

```python
nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
```

### 3-4. 모델 규모

| 항목 | 값 |
|---|---|
| 총 파라미터 수 | 31,037,698개 |
| 입력 | [B, 3, H, W] |
| 출력 | [B, num_classes, H, W] |
| 최소 입력 크기 | 16×16 (MaxPool 4회) |

---

## 4. 데이터셋 (`dataset.py`)

### 4-1. Oxford-IIIT Pet Dataset

- **규모**: 37개 품종, 약 7,349장
- **분할**: trainval(3,680장) / test(3,669장)
- **마스크**: 픽셀 단위 segmentation mask 포함
  - 1 = 반려동물(foreground)
  - 2 = 배경(background)
  - 3 = 경계선(boundary)

### 4-2. 전처리

**이미지**
```python
Resize(128×128)
→ ToTensor()           # [0,255] → [0.0,1.0]
→ Normalize(           # ImageNet 통계로 정규화
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225]
  )
```

**마스크**
```python
Resize(128×128, interpolation=NEAREST)  # 보간 없이 리사이즈 (클래스 값 보존)
→ (mask == 1).long()                    # 이진화: pet=1, 나머지=0
```

마스크 리사이즈에 `NEAREST` 보간을 사용하는 이유: 일반 보간(bilinear 등)을 적용하면 클래스 경계에 중간값이 생겨 레이블이 깨집니다.

### 4-3. 데이터 분할

```
전체 trainval (3,680장)
  → 무작위 800장 선택 (seed=42, 실험 속도 향상)
    → Train: 640장 (80%)
    → Val:   160장 (20%)

Test: 400장 (전체 3,669장 중 무작위 선택)
```

---

## 5. 학습 로직 (`train.py`)

### 5-1. 평가 메트릭

**Dice Coefficient**

의료 영상 분야의 표준 지표로, 예측과 정답의 겹치는 면적 비율입니다.

```
Dice = 2|A ∩ B| / (|A| + |B|)
```

**IoU (Intersection over Union / Jaccard Index)**

합집합 대비 교집합 비율로, Dice보다 엄격한 지표입니다.

```
IoU = |A ∩ B| / |A ∪ B|
```

**Pixel Accuracy**

```
Pixel Acc = 정확히 분류된 픽셀 수 / 전체 픽셀 수
```

세 메트릭 모두 `smooth=1e-6`을 분자/분모에 더해 분모가 0이 되는 경우를 방지합니다.

### 5-2. 손실 함수 (논문 Eq.1)

논문의 weighted cross entropy를 단순화하여 적용합니다.

```python
class_weights = torch.tensor([0.4, 1.0])  # [background, pet]
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

반려동물 픽셀(foreground)이 배경보다 적으므로 높은 가중치(1.0)를 부여합니다. 이는 논문 Eq.2의 `w(x)` weight map을 클래스 단위로 단순화한 것입니다.

### 5-3. 옵티마이저 (논문 Section 3)

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.99,    # 논문에서 명시한 값
    weight_decay=1e-4 # L2 정규화
)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
```

- `momentum=0.99`: 논문에서 명시. 일반적인 0.9보다 높아 과거 gradient를 더 강하게 반영
- `ReduceLROnPlateau`: Val Dice가 3 에포크 동안 개선되지 않으면 학습률을 절반으로 감소

### 5-4. 학습 루프

```
에포크마다:
  train_one_epoch()
    ├─ model.train()
    ├─ Forward → Loss → Backward → Step
    └─ Dice, IoU 계산

  evaluate()
    ├─ model.eval()  +  torch.no_grad()
    └─ Loss, Dice, IoU, PixelAcc 계산

  scheduler.step(val_dice)

  if val_dice > best_dice:
      → checkpoints/best_model.pth 저장
```

`@torch.no_grad()`: 평가 시 gradient 계산을 비활성화해 메모리와 속도를 절약합니다.
`model.eval()`: BatchNorm이 학습 통계 대신 저장된 이동 평균 통계를 사용하도록 전환합니다.

### 5-5. 학습 설정 요약

| 항목 | 값 |
|---|---|
| Epochs | 20 |
| Batch size | 16 |
| Image size | 128×128 |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Momentum | 0.99 (논문 기반) |
| Weight decay | 1e-4 |
| LR Scheduler | ReduceLROnPlateau (patience=3) |
| Loss | Weighted CrossEntropy |
| 디바이스 | Apple MPS (Metal GPU) |

---

## 6. 전체 데이터 흐름

```
Oxford Pet 이미지 (원본, 다양한 크기)
      │
      ▼ Resize(128×128) + Normalize(ImageNet 통계)
  이미지 텐서 [3, 128, 128]
  마스크 텐서 [128, 128]  (0 또는 1)
      │
      ▼ DataLoader (batch_size=16, shuffle=True)
  배치 [16, 3, 128, 128] / [16, 128, 128]
      │
      ▼ U-Net Forward Pass
  출력 [16, 2, 128, 128]  ← 채널 2: background / pet 로짓
      │
      ▼ CrossEntropyLoss (가중치 적용)
  Loss 스칼라
      │
      ▼ backward() + SGD.step()
  가중치 업데이트
      │
      ▼ softmax → [:, 1] (pet 확률) → threshold 0.5
  이진 마스크 [16, 128, 128]
      │
      ▼ Dice / IoU / PixelAcc
  성능 지표 출력
```

---

## 7. 실험 결과

### 7-1. 학습 과정

| Epoch | Train Loss | Val Loss | Train Dice | Val Dice | Val IoU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.7433 | 0.6234 | 0.4108 | 0.5420 | 0.3944 |
| 5 | 0.4148 | 0.4540 | 0.6850 | 0.6626 | 0.5189 |
| 10 | 0.2958 | 0.3806 | 0.7735 | 0.7397 | 0.6096 |
| 15 | 0.2234 | 0.3939 | 0.8133 | 0.7481 | 0.6213 |
| 20 | 0.1395 | 0.4564 | 0.8760 | 0.7677 | 0.6442 |

### 7-2. Test Set 최종 성능

| 메트릭 | 결과 |
|---|---|
| **Dice Coefficient** | **0.7692** |
| **IoU** | **0.6436** |
| **Pixel Accuracy** | **0.8662** |
| Loss | 0.3831 |

### 7-3. 결과 분석

- **Train Dice (0.88) vs Val Dice (0.77)**: 어느 정도의 과적합 경향이 있으나, test set에서도 0.77로 일관된 성능을 보임
- **에포크당 학습 시간**: 약 16초 (Apple MPS 기준)
- 에포크 초반(1~10)에 급격히 성능이 향상되고, 후반(10~20)에는 완만하게 수렴하는 전형적인 패턴

### 7-4. 결과 시각화

| 파일 | 내용 |
|---|---|
| `results/training_history.png` | Loss / Dice / IoU 학습 곡선 |
| `results/predictions.png` | 입력 이미지 / Ground Truth / U-Net 예측 비교 |

---

## 8. 실행 방법

### 환경 설정

```bash
pip install torch torchvision matplotlib
```

### 학습 실행

```bash
python train.py
```

학습 완료 후 `results/`, `checkpoints/` 폴더에 결과가 저장됩니다.

### 모델만 테스트

```bash
python unet_model.py
```

```
Input:  torch.Size([1, 3, 256, 256])
Output: torch.Size([1, 2, 256, 256])
총 파라미터 수: 31,037,698
```

### 데이터셋 확인

```bash
python dataset.py
```

```
Train: 640, Val: 160, Test: 400
이미지 shape: torch.Size([8, 3, 256, 256])
마스크 shape: torch.Size([8, 256, 256]), 고유값: tensor([0, 1])
```

---

## 참고 문헌

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*. arXiv:1505.04597
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers. arXiv:1502.01852
- Parkhi, O. M., et al. (2012). Cats and Dogs. *CVPR 2012*. (Oxford-IIIT Pet Dataset)
