# YOLOv1 — 단일 이미지 학습 예시

이 프로젝트는 **실제 데이터셋 없이** YOLOv1의 전체 학습·추론 파이프라인을 이미지 한 장으로 검증합니다.

---

## 파일 구조

```
yolov1/
├── model.py      ← YOLOv1 아키텍처 (LocallyConnected2d, ConvModule, YOLOv1)
├── loss.py       ← YOLOv1 Multi-part Loss (box / obj / noobj / cls)
├── dataset.py    ← SingleImageDataset, DemoSyntheticDataset
├── utils.py      ← IoU, NMS, encode_target, draw_boxes
├── train.py      ← 학습 스크립트 (CLI)
├── inference.py  ← 추론 스크립트 (CLI)
├── demo.py       ← 한 번에 전체 파이프라인 실행 (권장)
├── data/         ← 이미지 파일 놓는 곳
├── outputs/      ← 결과 이미지 저장
└── weights/      ← 체크포인트 저장
```

---

## 빠른 시작

### 1. 설치

```bash
pip install torch torchvision opencv-python matplotlib
```

### 2. 전체 파이프라인 한 번에 실행 (권장)

```bash
python demo.py
```

합성 이미지를 자동 생성해 학습 → 추론 → 결과 저장까지 수행합니다.

---

## 개별 스크립트 사용법

### 학습 (`train.py`)

```bash
# 합성 이미지로 학습 (이미지 파일 불필요)
python train.py

# 실제 이미지로 학습
python train.py --image data/my_photo.jpg

# 옵션
python train.py --image data/my_photo.jpg --epochs 100 --lr 1e-4 --batch 4
```

### 추론 (`inference.py`)

```bash
python inference.py --image data/my_photo.jpg --weights weights/best.pt
```

---

## 학습 방식

| 항목 | 내용 |
|---|---|
| 입력 해상도 | 448 × 448 |
| 그리드 크기 | S = 7 |
| 박스 개수 | B = 2 |
| 클래스 수 | C = 20 (PASCAL VOC) |
| 손실 함수 | YOLOv1 Multi-part Loss |
| 옵티마이저 | Adam + CosineAnnealingLR |
| Augmentation | HSV 색상 지터링 |

---

## 학습 데이터

단일 이미지를 반복(`repeat=200`)해 **과적합(overfit) 테스트**를 수행합니다.  
Loss 가 수렴하면 → 전체 파이프라인이 올바르게 작동함을 의미합니다.

실제 학습에는 PASCAL VOC 또는 COCO 데이터셋의 DataLoader를 `train.py`에 연결하세요.

---

## 출력 예시

```
weights/best.pt                  ← 가장 낮은 loss의 체크포인트
outputs/demo_input_with_gt.jpg   ← 입력 이미지 + GT 박스
outputs/demo_result.jpg          ← 모델 예측 결과
outputs/loss_curve.png           ← 에포크별 loss 그래프
```
