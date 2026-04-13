# U-Net Code Review & Implementation Notes
### Based on `hanyoseob/youtube-cnn-002-pytorch-unet`

---

## Overview

---

## Contents

- [1. What is U-Net?](#1-what-is-u-net)
- [2. Repository Structure](#2-repository-structure)
- [3. U-Net Core Idea](#3-u-net-core-idea)
- [4. Key Code Analysis](#4-key-code-analysis)
  - [4-1. Convolution Block](#4-1-convolution-block)
  - [4-2. Encoder](#4-2-encoder)
  - [4-3. Decoder](#4-3-decoder)
  - [4-4. Skip Connection](#4-4-skip-connection)
  - [4-5. Final Output Layer](#4-5-final-output-layer)
- [5. Dataset & Preprocessing](#5-dataset--preprocessing)
- [6. Loss Function & Training Loop](#6-loss-function--training-loop)
- [7. Test Output & Checkpoint](#7-test-output--checkpoint)
- [8. Summary](#8-summary)

---

## 1. What is U-Net?

U-Net은 **이미지 분할(Image Segmentation)** 을 위해 제안된 대표적인 Encoder-Decoder 구조의 CNN입니다.  
입력 이미지 전체를 받아, 각 픽셀마다 클래스를 예측하는 **dense prediction** 방식으로 동작합니다.

### U-Net의 핵심 특징

- **Encoder (Contracting Path)**  
  해상도를 줄이며 문맥(context) 정보를 추출
- **Decoder (Expansive Path)**  
  해상도를 복원하며 픽셀 단위 예측 수행
- **Skip Connection**  
  Encoder의 고해상도 feature를 Decoder에 직접 전달하여 위치 정보 보존

---

## 2. Repository Structure

이 저장소는 U-Net 구현에 필요한 핵심 파일들로 구성되어 있습니다.

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

각 파일의 역할
File	Role
data_read.py	원본 TIFF 데이터를 train / val / test용 .npy 파일로 분리
dataset.py	데이터 로딩, 정규화, augmentation, tensor 변환
model.py	U-Net 네트워크 구조 정의
train.py	학습 / 검증 / 테스트 루프 수행
util.py	체크포인트 저장 및 로드
eval.py	결과 평가용 보조 코드
display_results.py	결과 시각화
run_unet.ipynb	Notebook 환경에서 실행 예시
3. U-Net Core Idea

U-Net 논문의 핵심은 다음 세 문장으로 요약할 수 있습니다.

Encoder는 문맥 정보를 압축적으로 추출하고,
Decoder는 해상도를 복원하며,
Skip Connection은 위치 정보를 보존한다.

즉, U-Net은 단순히 이미지를 분류하는 모델이 아니라,
무엇이 있는지와 그것이 어디에 있는지를 동시에 학습하는 구조입니다.

4. Key Code Analysis
4-1. Convolution Block

U-Net에서 반복적으로 사용되는 기본 블록은 다음과 같은 구조입니다.

Conv2d
BatchNorm2d
ReLU
핵심 코드
def CBR2d(in_channels, out_channels):
    return Sequential(
        Conv2d(...),
        BatchNorm2d(...),
        ReLU()
    )
해설

이 블록은 feature를 추출하는 가장 기본적인 단위입니다.

Conv2d: 이미지의 지역 패턴 추출
BatchNorm2d: 학습 안정화
ReLU: 비선형성 부여

즉, U-Net의 각 단계에서 표현력 있는 feature map을 생성하는 기본 연산 단위라고 볼 수 있습니다.

4-2. Encoder

Encoder는 해상도를 줄이면서 채널 수를 늘려가는 구조입니다.

구조 흐름
64 → 128 → 256 → 512 → 1024
각 단계 사이에 MaxPool2d(kernel_size=2) 사용
핵심 코드
self.enc1_1, self.enc1_2
self.pool1
self.enc2_1, self.enc2_2
self.pool2
...
self.enc5_1
해설

Encoder의 목적은 단순히 feature map 크기를 줄이는 것이 아니라,
더 추상적이고 의미적인 표현을 학습하는 것입니다.

해상도 감소
채널 수 증가
semantic information 증가

즉, 얕은 층은 경계와 텍스처를 보고,
깊은 층은 더 고차원적인 구조를 이해하게 됩니다.

4-3. Decoder

Decoder는 줄어든 해상도를 다시 복원하는 역할을 합니다.

핵심 코드
self.unpool4 = nn.ConvTranspose2d(...)
self.dec4_2 = CBR2d(...)
self.dec4_1 = CBR2d(...)
해설

여기서 ConvTranspose2d는 feature map의 공간 크기를 키우는 역할을 합니다.
하지만 Decoder는 단순 업샘플링이 아니라,

Encoder에서 압축한 semantic information을 바탕으로
다시 픽셀 단위 위치 예측으로 되돌리는 과정

입니다.

즉,

Encoder가 무엇인가를 배우는 구조라면,
Decoder는 그것이 어디 있는가를 복원하는 구조입니다.

4-4. Skip Connection

U-Net의 가장 중요한 구조적 특징은 Skip Connection입니다.

핵심 코드
cat4 = torch.cat((unpool4, enc4_2), dim=1)
cat3 = torch.cat((unpool3, enc3_2), dim=1)
cat2 = torch.cat((unpool2, enc2_2), dim=1)
cat1 = torch.cat((unpool1, enc1_2), dim=1)
해설

깊은 층으로 갈수록 semantic information은 풍부해지지만,
반대로 세밀한 위치 정보(localization) 는 약해집니다.

이때 Skip Connection은

Encoder의 고해상도 feature를 보존하고
Decoder에 직접 전달하여
경계와 세부 구조를 정확하게 복원하게 만듭니다

즉, U-Net이 segmentation에 강한 핵심 이유는 바로 이 구조 때문입니다.

Semantic information + Localization information
두 가지를 동시에 유지하는 것이 U-Net의 핵심입니다.

4-5. Final Output Layer

최종 출력층은 1x1 convolution으로 구성됩니다.

핵심 코드
self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
해설

이 레이어는 각 픽셀 위치에서 feature를 종합해
최종적으로 foreground / background 여부를 예측합니다.

즉, 앞단에서 추출한 다채로운 feature map을
binary segmentation mask 로 변환하는 마지막 단계입니다.

5. Dataset & Preprocessing

이 저장소에서는 dataset.py에서 .npy 형식의 데이터를 불러와 전처리를 수행합니다.

핵심 코드
label = np.load(...)
input = np.load(...)
label = label / 255.0
input = input / 255.0
핵심 포인트
입력 영상과 정답 마스크를 각각 로드
0 ~ 255 값을 0 ~ 1 범위로 정규화
segmentation에서는 입력과 마스크의 정렬이 매우 중요
왜 중요한가?

classification에서는 이미지 한 장당 하나의 label이 있지만,
segmentation에서는 각 픽셀마다 정답이 존재합니다.

따라서

입력 이미지
정답 마스크
augmentation 결과

이 세 가지가 항상 정확히 대응되어야 합니다.

Data Augmentation

이 구현에서는 RandomFlip을 사용해 좌우/상하 반전을 적용합니다.

핵심 코드
if np.random.rand() > 0.5:
    label = np.fliplr(label)
    input = np.fliplr(input)
해설

segmentation에서는 augmentation을 할 때
입력과 label을 반드시 같은 방식으로 변환해야 합니다.

이 부분이 잘못되면

이미지와 마스크의 정렬이 깨지고
학습이 비정상적으로 진행되며
segmentation 품질이 크게 떨어집니다

즉, augmentation도 단순 보조 기능이 아니라
정합성 유지 측면에서 매우 중요한 구성 요소입니다.

6. Loss Function & Training Loop

이 저장소는 binary segmentation 문제를 다루기 때문에
손실 함수로 BCEWithLogitsLoss를 사용합니다.

핵심 코드
fn_loss = nn.BCEWithLogitsLoss().to(device)
해설

이 조합은 다음 의미를 가집니다.

출력 채널 수 = 1
각 픽셀마다 binary classification 수행
sigmoid + BCE를 수치적으로 안정적으로 결합한 loss 사용

즉, 이 모델은 한 장의 이미지 전체에 대해 픽셀 단위 이진 분류를 수행하는 구조입니다.

Training Loop
핵심 코드
output = net(input)
loss = fn_loss(output, label)
loss.backward()
optim.step()
해설

학습 루프 자체는 전형적인 PyTorch 구조이지만,
segmentation에서는 다음 점이 특히 중요합니다.

출력과 label의 shape가 정확히 일치해야 함
이미지 전체를 한 번에 forward
픽셀 전체에 대해 loss 계산

즉, 이 코드는 image-level classification 이 아니라
pixel-level prediction 을 수행하는 학습 구조입니다.

7. Test Output & Checkpoint

테스트 단계에서는 체크포인트를 불러오고,
예측 결과를 이미지와 numpy 형식으로 저장합니다.

핵심 코드
save(...)
load(...)
plt.imsave(...)
np.save(...)
해설

segmentation에서는 정량적 수치만 보는 것보다,
실제 예측 mask가 얼마나 자연스럽고 정확한지
직접 시각적으로 확인하는 것이 매우 중요합니다.

따라서 이 저장소는

checkpoint 저장 / 로드
예측 결과 이미지 저장
numpy 배열 저장

까지 포함해 실험 재현성과 시각적 검증을 모두 지원합니다.

8. Summary

이 구현은 복잡한 변형 모델이 아니라,
기본형 U-Net의 핵심 구조를 학습하기에 적절한 교육용 구현입니다.

핵심 요약
Encoder는 문맥 정보를 압축적으로 추출한다
Decoder는 해상도를 복원하며 픽셀 단위 예측을 수행한다
Skip Connection은 위치 정보를 보존해 segmentation 성능을 높인다
데이터 전처리와 입력-마스크 정합성이 매우 중요하다
최종적으로 이 구현은 binary image segmentation용 기본 U-Net 재현 예제라고 볼 수 있다
One-line Takeaway

이 코드는 U-Net 논문의 핵심 구조인 contracting path, expansive path, skip connection 을
PyTorch로 직관적으로 재현한 binary segmentation 교육용 구현 예제이다.

Reference
Original Repository: hanyoseob/youtube-cnn-002-pytorch-unet
Paper: U-Net: Convolutional Networks for Biomedical Image Segmentation
