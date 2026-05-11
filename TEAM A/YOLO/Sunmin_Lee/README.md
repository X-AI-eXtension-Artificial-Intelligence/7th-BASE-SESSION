# YOLO v1

> 본 레포지토리는 [nsoul97/yolov1_pytorch](https://github.com/nsoul97/yolov1_pytorch)를 Fork하여 논문 리뷰 내용을 추가한 학습용 레포지토리입니다.

---

## 📌 논문 정보

- **제목**: You Only Look Once: Unified, Real-Time Object Detection
- **저자**: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
- **발표**: CVPR 2015
- **논문 링크**: https://arxiv.org/abs/1506.02640

---

## 📖 논문 리뷰 요약

### 1. 배경 (Background)

객체 탐지(Object Detection)는 이미지 안에서 객체의 위치와 종류를 동시에 찾아내는 task로, 자율주행, 보안 카메라, 로보틱스 등 실시간 응용에 대한 수요가 급증하던 분야였다.

YOLO 이전의 주류는 **R-CNN 계열의 2-stage detector**였다.

| 모델 | 연도 | 핵심 개선 | 한계 |
|------|------|-----------|------|
| R-CNN | 2013 | CNN 기반 분류 도입 | Selective Search 의존, 수십 초 소요 |
| Fast R-CNN | 2015 | 전체 이미지 CNN 1회 적용 | 여전히 Selective Search 의존 |
| Faster R-CNN | 2015 | RPN 도입으로 E2E 구조 달성 | 2-stage 유지, 약 7 FPS |

기존 2-stage 방식의 근본적인 문제는 **후보 영역 추출(Proposal)과 분류(Classification)의 분리**였다. 아무리 최적화해도 속도에 구조적 한계가 존재했다.

---

### 2. 핵심 아이디어

> **"객체 탐지를 단일 신경망의 회귀(Regression) 문제로 재정의한다"**

```
기존: 이미지 → 후보 영역 추출 → 각 영역 분류  (2-stage)
YOLO: 이미지 → 그리드 분할 → 위치 + 클래스 동시 예측  (1-stage, E2E)
```

이미지를 **S×S 그리드**로 나누고, 각 셀에서 bounding box 좌표, confidence, class probability를 **하나의 벡터로 동시에 예측**한다.

---

### 3. 모델 구조 (Unified Detection)

**출력 텐서**

```
S × S × (B×5 + C)
= 7 × 7 × 30  (기본값)

7×7  : 그리드 셀 수 (448 입력을 6번 다운샘플링)
B×5  : bbox 2개 × [x, y, w, h, confidence]
C    : class 20개 (PASCAL VOC 기준)
```

**셀 하나의 출력 벡터 (30차원)**

```
[x, y, w, h, conf] [x, y, w, h, conf] [p1, p2, ... p20]
←   bbox 1 (5)   → ←   bbox 2 (5)   → ←  class (20)  →
```

**Confidence**

```
confidence = Pr(Object) × IoU
→ "객체가 있을 확률 × 박스 위치가 얼마나 정확한지"를 동시에 반영
```

**네트워크 구조**: GoogLeNet에서 영감을 받은 **24개 Conv + 2개 FC** 구조

```
입력: 448×448×3
→ Conv Block 1~6 (1×1 Conv로 채널 축소, 3×3 Conv로 피처 추출)
→ Flatten → FC(4096) → Dropout(0.5) → FC(1470)
→ 출력: 7×7×30
```

> **왜 FC layer를 쓰는가?**  
> 1×1 Conv는 각 위치를 독립적으로 처리하지만, FC layer는 feature map 전체를 펼쳐서 **이미지 전체의 글로벌 컨텍스트**를 참조할 수 있다. 이것이 "이미지 전체를 한 번에 본다"는 YOLO의 핵심 철학을 구현하는 방식이다. (YOLO v2부터는 FCN 구조로 대체)

---

### 4. Loss Function

```
Loss = 5   × (x,y 위치 오차)           ← 위치, 가중치 높음
     + 5   × (√w, √h 크기 오차)        ← 크기, 작은 박스 보정
     + 1   × (confidence 오차, 객체 O) ← 있는 곳 confidence
     + 0.5 × (confidence 오차, 객체 X) ← 없는 곳 가중치 낮춤
     + 1   × (class 오차)              ← 분류
```

| 설계 선택 | 이유 |
|-----------|------|
| λ_coord = 5 | 객체 없는 셀이 압도적으로 많아 위치 오차를 강조 |
| λ_noobj = 0.5 | 없는 셀의 loss 쏠림 방지 |
| w, h에 √ 적용 | 작은 박스의 오차를 상대적으로 크게 반영 |
| 담당 bbox 1개만 학습 | GT와 IoU 높은 bbox만 반영 |

---

### 5. 한계점 (Limitations)

| 한계 | 원인 |
|------|------|
| 밀집 객체 탐지 취약 | 셀당 bbox 2개 제한 |
| 다중 클래스 탐지 불가 | 셀당 class 1개 (bbox들이 class 공유) |
| Localization 정확도 낮음 | 그리드 단위 예측의 구조적 한계 |
| 작은 객체 탐지 취약 | √ 보정으로 완화했지만 근본 한계 존재 |

---

### 6. 실험 결과 (Experiments)

**PASCAL VOC 2007 성능 비교**

| 모델 | mAP | FPS |
|------|-----|-----|
| Faster R-CNN | 73.2% | 7 |
| Fast R-CNN | 70.0% | 0.5 |
| YOLO | 63.4% | 45 |
| Fast YOLO | 52.7% | 155 |

**YOLO + Fast R-CNN 앙상블**

```
Fast R-CNN 단독       : 70.0% mAP
YOLO 단독             : 63.4% mAP
YOLO + Fast R-CNN     : 75.0% mAP  ← SOTA 능가
```

두 모델의 오류 패턴이 상호 보완적이기 때문이다.
- Fast R-CNN: **Background 오탐** 많음 (맥락 정보 부족)
- YOLO: **Localization 오류** 많음 (그리드 구조 한계)

**일반화 성능 (Generalization)**

자연 사진으로 학습 후 예술 작품(피카소 그림 등)에 테스트했을 때, YOLO가 R-CNN보다 도메인 변화에 더 강건한 성능을 보였다. 이미지 전체의 글로벌 피처를 학습하기 때문이다.

---

### 7. 의의 (Conclusion)

> YOLO의 진짜 기여는 단순히 빠른 모델을 만든 것이 아니라,  
> **"탐지 = 복잡한 파이프라인"이라는 고정관념을 깨고 단일 회귀 문제로 통합한 새로운 관점을 제시한 것**이다.  
> 이후 모든 1-stage detector(SSD, RetinaNet, YOLO v2~v8)가 이 관점에서 출발한다.

---

## 🔧 Requirements

```
torch
torchvision
matplotlib
pillow
tqdm
```

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

### PASCAL VOC 2007 + 2012

```bash
./scripts/download_voc.sh
./scripts/organize_voc.sh
python3 code/simplify_voc_targets.py
```

### ImageNet 2012 (사전학습용)

[ImageNet 공식 사이트](https://image-net.org/)에서 회원가입 후 아래 파일 다운로드:
- `ILSVRC2012_img_train.tar`
- `ILSVRC2012_img_val.tar`
- `ILSVRC2012_devkit_t12.tar.gz`

```bash
./scripts/organize_imagenet.sh
```

---

## 🚀 실행 방법

**사전학습 모델 평가**
```bash
python3 code/pretrain.py
```

**YOLO 탐지 성능 평가**
```bash
python3 code/evaluate.py
```

**예측 결과 시각화**
```bash
python3 code/plot_predictions.py
```

---

## 📊 결과

| Implementation | Mean Average Precision |
|----------------|----------------------|
| this repository | 63.6% |
| paper | 63.4% |




## 📝 원본 구현과의 차이점

본 레포는 공식 [Darknet](https://github.com/pjreddie/darknet) 구현을 따르며, 논문과 다음과 같은 차이가 있다:

1. **첫 번째 FC layer → Locally Connected Layer로 대체**
2. **각 Conv layer에 Batch Normalization 추가** (Conv → BN → Activation)
3. **학습률 스케줄 및 max_batches 조정**

---

## 📚 References

- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2015). You Only Look Once: Unified, Real-Time Object Detection. CVPR 2015.
- Everingham, M. et al. (2014). The Pascal Visual Object Classes Challenge: A Retrospective. IJCV.
- 원본 구현 레포: https://github.com/nsoul97/yolov1_pytorch
