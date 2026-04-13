````md
# U-Net Code Review & Implementation Notes
### Based on `hanyoseob/youtube-cnn-002-pytorch-unet`

---

## Overview


분석 대상은 `hanyoseob/youtube-cnn-002-pytorch-unet` 저장소이며, 데이터 전처리부터 모델 정의, 학습, 테스트까지의 전체 흐름을 기준으로 정리하였다.  
특히 본 문서에서는 U-Net의 핵심 구성 요소인 **Encoder, Decoder, Skip Connection, Output Layer**를 중심으로 실제 코드가 어떤 방식으로 이를 구현하고 있는지 단계적으로 설명한다.

---

## Contents

- [1. What is U-Net?](#1-what-is-u-net)
- [2. Repository Structure](#2-repository-structure)
- [3. Overall Pipeline](#3-overall-pipeline)
- [4. U-Net Core Architecture](#4-u-net-core-architecture)
  - [4-1. Basic Convolution Block](#4-1-basic-convolution-block)
  - [4-2. Encoder](#4-2-encoder)
  - [4-3. Decoder](#4-3-decoder)
  - [4-4. Skip Connection](#4-4-skip-connection)
  - [4-5. Final Output Layer](#4-5-final-output-layer)
- [5. Dataset and Preprocessing](#5-dataset-and-preprocessing)
- [6. Loss Function and Training Process](#6-loss-function-and-training-process)
- [7. Test and Result Saving](#7-test-and-result-saving)
- [8. Interpretation from the Paper Perspective](#8-interpretation-from-the-paper-perspective)
- [9. Summary](#9-summary)

---

## 1. What is U-Net?

U-Net은 **이미지 분할(Image Segmentation)** 문제를 해결하기 위해 제안된 대표적인 Encoder-Decoder 구조의 합성곱 신경망이다.  
분류 문제와 달리 segmentation은 이미지 전체에 대해 하나의 label을 예측하는 것이 아니라, **각 픽셀 단위로 클래스를 예측**해야 한다.  
따라서 모델은 단순히 "무엇이 존재하는가"를 판단하는 수준을 넘어, **그 구조가 정확히 어디에 존재하는가**까지 함께 추론해야 한다.

U-Net은 이러한 요구를 충족하기 위해 다음과 같은 구조를 갖는다.

- **Encoder (Contracting Path)**: 해상도를 줄이면서 문맥 정보를 추출
- **Decoder (Expansive Path)**: 해상도를 복원하면서 픽셀 단위 예측 수행
- **Skip Connection**: 얕은 층의 위치 정보를 깊은 층의 의미 정보와 결합

즉, U-Net은 **semantic information**과 **localization information**을 동시에 유지하는 구조라고 볼 수 있다.

---

## 2. Repository Structure

본 저장소는 U-Net 구현에 필요한 핵심 파일들로 구성되어 있으며, 각 파일은 다음과 같은 역할을 수행한다.

```text
.
├── data_read.py
├── dataset.py
├── model.py
├── train.py
├── util.py
├── eval.py
├── display_results.py
├── run_unet.ipynb
└── datasets/
````

### File Description

| File                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `data_read.py`       | 원본 TIFF 데이터를 읽어 train / val / test용 `.npy` 파일로 분할하는 전처리 스크립트 |
| `dataset.py`         | `.npy` 데이터 로딩, 정규화, augmentation, tensor 변환을 수행하는 데이터셋 정의    |
| `model.py`           | U-Net 네트워크 구조를 정의하는 핵심 모듈                                    |
| `train.py`           | 학습, 검증, 테스트 루프를 포함하는 실행 파일                                   |
| `util.py`            | 체크포인트 저장 및 로드 기능을 담당하는 보조 모듈                                 |
| `eval.py`            | 결과 평가를 위한 보조 스크립트                                            |
| `display_results.py` | 예측 결과를 시각화하는 스크립트                                            |
| `run_unet.ipynb`     | Notebook 환경에서의 실행 예시                                         |

이와 같이 저장소는 **데이터 구성 → 모델 정의 → 학습 및 테스트 → 결과 확인**의 전형적인 실험 파이프라인을 따르고 있다.

---

## 3. Overall Pipeline

본 구현의 전체 흐름은 다음과 같이 요약할 수 있다.

```text
Raw TIFF Data
   ↓
data_read.py
   ↓
Train / Val / Test .npy files
   ↓
dataset.py
   ↓
DataLoader
   ↓
model.py (U-Net)
   ↓
train.py
   ↓
Checkpoint / Prediction Results
```

이 구조는 단순히 모델만 정의하는 수준이 아니라, 실제 segmentation 실험에 필요한 전체 과정을 일관된 형태로 포함하고 있다는 점에서 교육용 구현으로서 의미가 있다.

---

## 4. U-Net Core Architecture

U-Net 구조의 핵심은 다음과 같다.

1. 합성곱 블록을 반복하며 특징을 추출한다.
2. Encoder에서 해상도를 줄이며 의미 정보를 축적한다.
3. Decoder에서 해상도를 복원하며 segmentation mask를 생성한다.
4. Skip Connection을 통해 Encoder의 세밀한 위치 정보를 Decoder로 전달한다.

이하에서는 이를 코드 수준에서 단계적으로 설명한다.

---

### 4-1. Basic Convolution Block

본 구현에서는 기본적인 feature extraction 단위를 `CBR2d`로 정의하고 있다.

```python
def CBR2d(in_channels, out_channels):
    layers = []
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=3, stride=1, padding=1, bias=True)]
    layers += [nn.BatchNorm2d(num_features=out_channels)]
    layers += [nn.ReLU()]

    cbr = nn.Sequential(*layers)

    return cbr
```

#### Interpretation

해당 블록은 **Conv2d → BatchNorm2d → ReLU** 구조로 이루어져 있다.
이는 현대 CNN에서 매우 표준적인 구성으로, 각 요소는 다음 역할을 수행한다.

* **Conv2d**: 입력 이미지로부터 지역적 패턴과 형태 정보를 추출
* **BatchNorm2d**: feature distribution을 안정화하여 학습을 원활하게 함
* **ReLU**: 비선형성을 부여하여 복잡한 표현 학습 가능

즉, 이 블록은 U-Net 전체 구조를 이루는 가장 기본적인 feature extraction unit이며, Encoder와 Decoder의 대부분의 층에서 반복적으로 사용된다.

---

### 4-2. Encoder

Encoder는 입력 이미지의 공간 해상도를 점차 줄이면서 더 고차원적인 feature를 추출하는 역할을 한다.

```python
self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
self.pool1 = nn.MaxPool2d(kernel_size=2)

self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
self.pool2 = nn.MaxPool2d(kernel_size=2)

self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
self.enc3_2 = CBR2d(in_channels=256, out_channels=256)
self.pool3 = nn.MaxPool2d(kernel_size=2)

self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
self.enc4_2 = CBR2d(in_channels=512, out_channels=512)
self.pool4 = nn.MaxPool2d(kernel_size=2)

self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
```

#### Interpretation

Encoder에서는 다음과 같은 패턴이 반복된다.

* 해상도는 `MaxPool2d`를 통해 절반으로 감소
* 채널 수는 `64 → 128 → 256 → 512 → 1024`로 증가

이는 U-Net 논문에서 제안된 contracting path의 전형적인 형태와 일치한다.

이 구조가 중요한 이유는 다음과 같다.

* 얕은 층에서는 **경계, 텍스처, 저수준 시각 특징**을 포착
* 깊은 층에서는 **객체 구조, 의미 정보, 고수준 표현**을 학습

즉, Encoder는 단순히 feature map 크기를 줄이는 과정이 아니라, **입력을 점차 더 추상적이고 의미 있는 표현으로 변환하는 과정**으로 해석할 수 있다.

---

### 4-3. Decoder

Decoder는 Encoder에서 축소된 해상도를 다시 복원하면서 segmentation mask 형태의 출력을 생성하는 역할을 한다.

```python
self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                  kernel_size=2, stride=2, padding=0, bias=True)
self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                  kernel_size=2, stride=2, padding=0, bias=True)
self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                  kernel_size=2, stride=2, padding=0, bias=True)
self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                  kernel_size=2, stride=2, padding=0, bias=True)
self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
self.dec1_1 = CBR2d(in_channels=64, out_channels=64)
```

#### Interpretation

Decoder에서는 `ConvTranspose2d`를 사용하여 해상도를 다시 증가시킨다.
그러나 이 단계는 단순한 업샘플링이 아니라, **Encoder에서 학습된 고수준 의미 표현을 다시 픽셀 수준의 예측 공간으로 복원하는 과정**이다.

즉, Decoder는 다음과 같은 역할을 한다.

* 축소된 feature map을 다시 원래 크기에 가깝게 복원
* segmentation에 필요한 spatial detail을 점차 회복
* 최종적으로 각 픽셀에 대한 예측 mask 생성

논문 관점에서 보면, Decoder는 Encoder의 반대 방향으로 진행되지만 단순 역연산이 아니라, **feature refinement와 localization recovery를 수행하는 모듈**이다.

---

### 4-4. Skip Connection

U-Net의 가장 핵심적인 구조적 장점은 Skip Connection에 있다.

```python
cat4 = torch.cat((unpool4, enc4_2), dim=1)
cat3 = torch.cat((unpool3, enc3_2), dim=1)
cat2 = torch.cat((unpool2, enc2_2), dim=1)
cat1 = torch.cat((unpool1, enc1_2), dim=1)
```

#### Interpretation

Encoder가 깊어질수록 의미 정보는 풍부해지지만, 반대로 **세밀한 위치 정보는 약해지는 경향**이 있다.
이는 pooling을 반복하는 구조의 필연적인 특징이다.

이를 보완하기 위해 U-Net은 동일한 해상도를 가지는 Encoder의 feature map을 Decoder에 직접 연결한다.
코드에서는 이를 `torch.cat(..., dim=1)`으로 구현하고 있다.

이 구조의 의의는 다음과 같다.

* Encoder의 고해상도 정보를 유지
* Decoder의 복원 과정에서 경계 및 세부 구조를 정밀하게 보존
* 깊은 의미 정보와 얕은 위치 정보를 동시에 활용

따라서 Skip Connection은 U-Net이 일반적인 Encoder-Decoder보다 segmentation에서 더 우수한 성능을 보이는 핵심 요인으로 해석된다.

---

### 4-5. Final Output Layer

최종 출력층은 1x1 convolution으로 구성된다.

```python
self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
```

#### Interpretation

`1x1 convolution`은 공간 크기를 변경하지 않으면서 각 위치의 채널 정보를 통합하여 최종 출력을 생성하는 역할을 한다.
본 구현에서는 출력 채널 수가 1이므로, 이는 **각 픽셀에 대해 하나의 binary logit 값을 출력**한다는 의미이다.

즉, 최종 출력층은 다음 역할을 수행한다.

* Decoder에서 생성된 feature map을 segmentation mask 형태로 변환
* foreground / background에 대한 픽셀 단위 점수 생성

이 구조는 binary segmentation 문제에 적합하며, 이후 BCE 기반 손실함수와 자연스럽게 연결된다.

---

## 5. Dataset and Preprocessing

본 저장소에서는 `dataset.py`를 통해 `.npy` 데이터를 불러오고, 기본적인 전처리 및 augmentation을 수행한다.

### Data Loading

```python
label = np.load(os.path.join(self.data_dir, self.label_lst[index]))
input = np.load(os.path.join(self.data_dir, self.input_lst[index]))
```

#### Interpretation

전처리된 데이터는 `label_XXX.npy`, `input_XXX.npy` 형식으로 저장되어 있으며, 데이터셋 클래스는 이를 각각 정답 마스크와 입력 영상으로 불러온다.

이는 segmentation 문제의 특성상 입력과 label이 항상 **1:1로 정확히 대응**되어야 함을 반영한다.

---

### Normalization

```python
label = label / 255.0
input = input / 255.0
```

#### Interpretation

원본 픽셀 값 범위를 `0 ~ 255`에서 `0 ~ 1`로 스케일링하는 과정이다.
이는 수치적 안정성을 높이고 학습을 원활하게 하기 위한 가장 기본적인 전처리 단계이다.

특히 segmentation에서는 입력뿐 아니라 label도 일관된 범위로 유지되는 것이 중요하다.

---

### Data Augmentation

```python
if np.random.rand() > 0.5:
    label = np.fliplr(label)
    input = np.fliplr(input)

if np.random.rand() > 0.5:
    label = np.flipud(label)
    input = np.flipud(input)
```

#### Interpretation

본 구현에서는 좌우 반전과 상하 반전을 적용하여 데이터 다양성을 확보한다.
여기서 중요한 점은 **입력 이미지와 정답 마스크에 동일한 augmentation이 동시에 적용되어야 한다는 것**이다.

만약 입력과 label에 서로 다른 변환이 적용되면 픽셀 정렬이 깨지므로, segmentation 학습 자체가 성립하지 않게 된다.
따라서 해당 augmentation 코드는 단순한 성능 향상 수단을 넘어서, **데이터 정합성을 유지하는 핵심 구현 요소**로 볼 수 있다.

---

## 6. Loss Function and Training Process

### Loss Function

본 구현은 binary segmentation 문제를 대상으로 하므로 `BCEWithLogitsLoss`를 사용한다.

```python
fn_loss = nn.BCEWithLogitsLoss().to(device)
```

#### Interpretation

이 손실함수는 다음과 같은 이유로 적절하다.

* 출력 채널이 1개인 binary segmentation 구조와 자연스럽게 대응
* sigmoid와 binary cross entropy를 수치적으로 안정적인 형태로 결합
* 각 픽셀을 독립적인 binary classification 문제로 처리 가능

즉, 본 구현은 **"출력 1채널 + BCEWithLogitsLoss"** 조합을 통해 binary segmentation 문제를 효과적으로 다루고 있다.

---

### Training Loop

```python
output = net(input)
loss = fn_loss(output, label)

optim.zero_grad()
loss.backward()
optim.step()
```

#### Interpretation

학습 루프는 전형적인 PyTorch 구조를 따른다.
그러나 segmentation 관점에서 특히 중요한 점은 다음과 같다.

* 입력 이미지 전체가 네트워크에 들어감
* 출력은 이미지 전체에 대한 픽셀 단위 예측 map임
* loss는 각 픽셀 위치에 대해 계산됨

즉, 이 코드는 일반적인 image classification이 아니라, **dense prediction을 위한 학습 구조**를 구현하고 있다.

---

## 7. Test and Result Saving

테스트 단계에서는 학습된 체크포인트를 불러와 예측 결과를 저장한다.

### Checkpoint Handling

```python
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
```

#### Interpretation

학습된 모델 파라미터를 불러와 이어서 평가 또는 추가 학습이 가능하도록 한다.
이는 실험 재현성과 학습 안정성 측면에서 매우 중요한 요소이다.

---

### Saving Prediction Results

```python
plt.imsave(os.path.join(result_dir, "png", "input_%04d.png" % id), input_arr, cmap='gray')
plt.imsave(os.path.join(result_dir, "png", "label_%04d.png" % id), label_arr, cmap='gray')
plt.imsave(os.path.join(result_dir, "png", "output_%04d.png" % id), output_arr, cmap='gray')

np.save(os.path.join(result_dir, "numpy", "input_%04d.npy" % id), input_arr)
np.save(os.path.join(result_dir, "numpy", "label_%04d.npy" % id), label_arr)
np.save(os.path.join(result_dir, "numpy", "output_%04d.npy" % id), output_arr)
```

#### Interpretation

Segmentation 문제에서는 단순 수치 지표뿐 아니라, 실제 예측 mask가 얼마나 자연스럽고 정확한지를 시각적으로 확인하는 과정이 매우 중요하다.
본 구현은 이를 위해 결과를 `png`와 `npy` 두 형태로 모두 저장한다.

* `png`: 시각적 확인 용도
* `npy`: 후속 분석 및 정량 평가 용도

이는 논문 실험에서도 흔히 사용되는 **qualitative + quantitative evaluation** 흐름과 일치한다.

---

## 8. Interpretation from the Paper Perspective

본 구현은 복잡한 변형 U-Net이 아니라, **기본형 U-Net의 핵심 철학을 비교적 충실하고 직관적으로 구현한 예제**라고 평가할 수 있다.

논문 관점에서 해석하면 다음과 같다.

* Encoder는 문맥 정보를 축적하는 역할을 수행한다.
* Decoder는 축소된 표현을 다시 공간적으로 복원한다.
* Skip Connection은 localization 성능을 유지하는 핵심 장치로 작동한다.
* 최종 출력층은 binary segmentation mask를 생성한다.
* 데이터 전처리와 augmentation은 segmentation 정합성을 보장한다.

즉, 본 저장소는 **U-Net 논문의 구조적 핵심이 실제 코드에서 어떤 형태로 구현되는지 학습하기에 적합한 교육용 예제**라고 볼 수 있다.

---

## 9. Summary

본 구현은 다음과 같은 점에서 의미가 있다.

* 기본형 U-Net 구조를 직관적으로 이해할 수 있음
* segmentation을 위한 전체 실험 파이프라인을 포함함
* 논문 구조와 코드 구현 간 대응 관계를 명확히 확인할 수 있음
* 데이터 전처리, 모델 정의, 학습, 테스트의 흐름이 일관되게 정리되어 있음

### Final Takeaway

> 이 구현은 U-Net 논문의 핵심 구조인 **contracting path, expansive path, skip connection**을
> PyTorch로 충실하게 재현한 **binary image segmentation 교육용 구현 예제**로 해석할 수 있다.

---

## Reference

* Original Repository: `hanyoseob/youtube-cnn-002-pytorch-unet`
* Paper: **U-Net: Convolutional Networks for Biomedical Image Segmentation**

```
```
