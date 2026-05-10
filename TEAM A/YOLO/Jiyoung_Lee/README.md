# YOLOv1 PyTorch Implementation

YOLOv1(You Only Look Once) 논문을 PyTorch로 직접 구현한 프로젝트입니다.

> Paper:  
> https://arxiv.org/abs/1506.02640

본 프로젝트는:

- ImageNet Backbone Pretraining
- VOC Detection Fine-tuning
- YOLO Loss 직접 구현
- Data Augmentation 직접 구현
- mAP Evaluation
- Bounding Box Visualization

까지 포함하여 YOLOv1 전체 파이프라인을 구현했습니다.

---

# Project Structure

```bash
.
├── dataset.py
├── transforms.py
├── model.py
├── loss.py
├── train.py
├── pretrain.py
├── evaluate.py
├── plot_predictions.py
├── checkpoints/
├── assets/
└── README.md
```

---

# File Description

| File | Description |
|---|---|
| dataset.py | VOC Dataset Loader |
| transforms.py | Data Augmentation + YOLO Tensor 변환 |
| model.py | YOLOv1 모델 구현 |
| loss.py | YOLO Loss + IOU 계산 |
| train.py | Detection 학습 |
| pretrain.py | ImageNet Classification Pretraining |
| evaluate.py | mAP Evaluation |
| plot_predictions.py | Bounding Box Visualization |

---

# YOLOv1 Overview

YOLOv1은 이미지를 S × S grid로 분할하고,

각 grid cell마다:

- Class Probability
- Objectness Confidence
- Bounding Box Coordinates

를 동시에 예측하는 Object Detection 모델입니다.

---

# Model Architecture

현재 구현 특징:

- Conv + BatchNorm + LeakyReLU 구조
- Locally Connected Layer 사용
- Gradient Accumulation 적용
- Burn-in Learning Rate Scheduler 적용
- Multi-Step LR Scheduler 적용

---

# Dataset

## 1. PASCAL VOC Dataset

사용 Dataset:

- VOC2007 train/val
- VOC2012 train/val
- VOC2007 test

### Dataset Download

```bash
./download_voc.sh
./organize_voc.sh

python3 simplify_voc_targets.py
```

---

## 2. ImageNet 2012 Dataset

YOLO Backbone Pretraining에 사용됩니다.

### Required Files

- ILSVRC2012_img_train.tar
- ILSVRC2012_img_val.tar
- ILSVRC2012_devkit_t12.tar.gz

### Dataset Organizing

```bash
./organize_imagenet.sh
```

---

# Requirements

```txt
torch
torchvision
matplotlib
pillow
tqdm
```

### Install

```bash
pip install -r requirements.txt
```

---

# Training

## 1. ImageNet Pretraining

Backbone Classification Pretraining

### Run

```bash
python3 pretrain.py
```

### Features

- CrossEntropyLoss
- SGD + Momentum
- LR Scheduler
- Top1 / Top5 Accuracy Evaluation

---

## 2. YOLO Detection Training

Detection Fine-tuning

### Run

```bash
python3 train.py
```

### Features

- YOLO Loss 직접 구현
- IOU 기반 Responsible Box 선택
- Gradient Accumulation
- Burn-in Learning Rate
- Multi-Step LR Scheduler
- Custom Data Augmentation

---

# Data Augmentation

`transforms.py`에서 직접 구현:

- Resize
- Random Scale & Translate
- Random Zoom In
- Random Zoom Out
- Random HSV Distortion
- Random Horizontal Flip

---

# Evaluation

mAP 계산:

```bash
python3 evaluate.py
```

### Evaluation Pipeline

1. Bounding Box Rescaling
2. Confidence Filtering
3. Non-Max Suppression
4. IOU Matching
5. Average Precision Calculation
6. Mean Average Precision Calculation

---

# Visualization

Prediction Visualization:

```bash
python3 plot_predictions.py
```

### Key Controls

| Key | Function |
|---|---|
| Left Arrow | 이전 이미지 |
| Right Arrow | 다음 이미지 |
| S | 현재 이미지 저장 |
| Q | 종료 |

---

# YOLO Loss

현재 구현된 YOLO Loss:

## 1. Localization Loss

Bounding Box 위치 오차 계산

- center x/y
- width/height

---

## 2. Objectness Loss

Object 존재 여부 confidence 계산

---

## 3. Classification Loss

Class prediction 오차 계산

---

# Important Implementation Details

## Locally Connected Layer

논문 Fully Connected Layer 대신:

- 위치별 다른 weight 사용
- spatial information 유지

를 위해 Locally Connected Layer 사용

---

## Responsible Bounding Box Selection

각 grid cell의 B개 bbox 중:

- 가장 높은 IOU bbox 선택
- IOU가 모두 0이면 RMSE 기준 선택

---

## Gradient Accumulation

```python
loss = loss / SUBDIVISIONS
loss.backward()
```

메모리 사용량 감소를 위해 Gradient Accumulation 사용

---

# Results

## ImageNet Pretraining

| Metric | Result |
|---|---|
| Top5 Accuracy | 89% |

논문 성능:
- 88%

---

## VOC Detection

| Implementation | mAP |
|---|---|
| This Repository | 63.6% |
| Original Paper | 63.4% |

---

# Prediction Examples

- Person Detection
- Vehicle Detection
- Animal Detection
- Multiple Object Detection

Bounding Box 위 숫자는 Objectness Confidence입니다.

---

# Run Summary

## Pretraining

```bash
python3 pretrain.py
```

## Detection Training

```bash
python3 train.py
```

## Evaluation

```bash
python3 evaluate.py
```

## Visualization

```bash
python3 plot_predictions.py
```

---

# References

- Joseph Redmon et al.  
  You Only Look Once: Unified, Real-Time Object Detection

- The Pascal Visual Object Classes Challenge

- Darknet Official Repository  
  https://github.com/pjreddie/darknet

- PyTorch Official Website  
  https://pytorch.org