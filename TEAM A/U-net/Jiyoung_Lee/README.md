# U-Net 기반 이미지 세그멘테이션 구현

---

## 1. Overview

일반적인 이미지 분류는 이미지 전체에 대해 하나의 label을 예측하지만,
세그멘테이션은 이미지 내의 **각 픽셀 단위로 클래스를 예측하는 문제**이다.
---

## 2. Repository Structure

```text id="repo_struct_v2"
.
├── data_read.py        # raw tif 데이터 → npy 변환 및 데이터 분할
├── dataset.py          # 데이터 로딩 및 전처리
├── model.py            # U-Net 모델 정의
├── train.py            # 학습 및 평가 루프
├── saved_model/        # 학습된 모델 가중치 저장
└── result/             # 예측 결과 저장

result/
 ├── numpy/   # 예측 결과 (.npy)
 └── png/     # 시각화 이미지
```

---

## 3. Overall Pipeline

전체 학습 과정은 다음과 같은 흐름으로 진행된다.

```text id="pipeline_v2"
Raw TIFF Data 
   ↓
data_read.py: raw TIFF 데이터를 불러옴 
   ↓
Train / Validation / Test (.npy)
   ↓
dataset.py (DataLoader):데이터 로드, 전처리 실행 후 모델에 입력 
   ↓
model.py (U-Net): 정의한 모델 사용해 학습 진행 
   ↓
train.py (Training / Evaluation): 학습과 동시에 결과 저장 
   ↓
Result Visualization
```

이 파이프라인은 데이터 전처리부터 학습, 결과 확인까지의 전체 과정을 포함한다.

---

## 4. U-Net Architecture

U-Net은 Contracting Path-Expansive Path 구조를 가지는 모델이다. 

### 4.1 Contracting Path

Contracting Path는 입력 이미지의 해상도를 점차 줄이면서
더 높은 수준의 feature를 추출한다.

* MaxPooling → 해상도 감소
* 채널 수 증가 (64 → 128 → 256 → 512 → 1024)

---

### 4.2 Expansive Path

Expansive Path는 Contracting Pathh에서 압축된 feature를
다시 원래 크기로 복원해 필섹 단위 예측 수행 

* ConvTranspose2d를 사용한 upsampling
* 해상도 복원
* 
---

### 4.3 Skip Connection (핵심 구조)

Contracting Path에서 downsampling이 반복되면
위치 정보가 손실된다.

이를 해결하기 위해 Contracting Path의 feature를
Expansive Path에 직접 연결(concatenate)한다.

```python id="skip_example_v2"
cat = torch.cat((decoder_feature, encoder_feature), dim=1)
```

* 위치 정보 보존
* 경계 정보 유지
* segmentation 정확도 향상

---

## 5. Dataset and Preprocessing

### 5.1 data_read.py

* multi-frame TIFF 데이터를 불러옴
* 데이터를 train / validation / test로 분할
* `.npy` 형태로 저장

목적

* 데이터 로딩 속도 향상
* 학습 효율 개선

---

### 5.2 dataset.py

데이터셋 클래스는 다음 기능을 수행한다.

#### (1) 데이터 로드

```python id="load_v2"
np.load(...)
```

* input / label 쌍을 함께 로드
* segmentation에서는 두 데이터의 정합성이 중요

---

#### (2) 정규화

```python id="norm_v2"
input = input / 255.0
```

* 픽셀 값을 [0,1] 범위로 변환
* 학습 안정성 향상

---

#### (3) 데이터 증강

```python id="aug_v2"
좌우 / 상하 flip
```

---

#### (4) Tensor 변환

```python id="tensor_v2"
(H, W, C) → (C, H, W)
```

---

## 6. Model Implementation

모델은 논문의 구조를 기반으로 구현되었다.

### 구성 요소

* Conv → BatchNorm → ReLU
* Encoder / Decoder 구조
* Skip Connection

---

### Output Layer

```python id="output_v2"
1x1 convolution
```

의미

* 각 픽셀에 대해 binary logit 출력
* segmentation mask 생성

---

## 7. Training Strategy

### 7.1 Loss Function

```python id="loss_v2"
BCEWithLogitsLoss
```

* 픽셀 단위 binary classification 수행

---

### 7.2 Optimizer

```python id="opt_v2"
Adam optimizer
```

* 빠르고 안정적인 학습

---

### 7.3 Training Flow

```text id="loop_v2"
Input → Model → Output → Loss → Backpropagation → Weight Update
```

---

## 8. Results

모델의 출력 결과는 다음과 같이 저장된다.

* input image
* ground truth label
* predicted output

---

## 9. Interpretation

본 구현은 U-Net의 핵심 아이디어를 그대로 반영한다.

* Contracting Path → semantic 정보 학습
* Expansive Path → spatial 정보 복원
* Skip Connection → 정보 결합

즉, 단순 CNN이 아니라
**의미 + 위치 정보를 동시에 활용하는 구조**이다.

---

## 10. Conclusion

본 프로젝트를 통해 U-Net 구조를 직접 구현하고
세그멘테이션 문제에 적용하였다.

Encoder-Decoder 구조와 Skip Connection을 통해
이미지의 의미 정보와 위치 정보를 동시에 활용할 수 있으며,
이는 픽셀 단위 예측 성능 향상에 중요한 역할을 한다.

---

## 11. Reference

* U-Net: Convolutional Networks for Biomedical Image Segmentation
* Ronneberger et al., 2015

---
