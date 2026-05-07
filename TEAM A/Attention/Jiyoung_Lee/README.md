# Seq2Seq + Attention 기반 기계번역 구현 (PyTorch)

---

## 1. Overview

본 프로젝트는 **Seq2Seq(Sequence-to-Sequence) 모델에 Attention 메커니즘을 적용하여**
영어 문장을 프랑스어 문장으로 번역하는 모델을 구현하는 것을 목표로 한다.

기존 Seq2Seq 모델은 입력 문장을 하나의 벡터로 압축하는 과정에서 정보 손실이 발생한다.
이를 해결하기 위해 **Attention을 사용하여 디코더가 필요한 시점에 인코더의 특정 부분을 집중적으로 참조하도록 설계하였다.**

---

## 2. Repository Structure

```
.
├── load_data.py        # 데이터 로드 및 전처리
├── model.py            # Encoder / Decoder / Attention 정의
├── train.py            # 학습 루프 및 실행 코드
├── data-2/
│   ├── eng-fra.txt     # 영어-프랑스어 번역 데이터셋
│   └── names/          # (사용하지 않음)
└── README.md
```

---

## 3. Dataset

본 프로젝트에서는 PyTorch 공식 튜토리얼에서 제공하는
**English-French Translation Dataset**을 사용하였다.

```
eng-fra.txt
```

데이터는 다음과 같은 형태로 구성된다.

```
i am hungry    →    je suis faim
he is tall     →    il est grand
```

---

## 4. Data Preprocessing

### 4.1 토큰화 및 Vocabulary 생성

* 문장을 단어 단위로 분리
* 각 단어를 index로 매핑

```python
word → index
```

---

### 4.2 Special Token

| Token   | 의미      |
| ------- | ------- |
| `<pad>` | padding |
| `<sos>` | 시작      |
| `<eos>` | 종료      |

---

### 4.3 Tensor 변환

```python
문장 → index sequence → tensor
```

예시:

```
i am hungry
→ [4, 7, 10, <eos>]
```

---

## 5. Model Architecture

전체 모델은 다음과 같은 구조를 가진다.

```
Input Sentence
      ↓
Encoder (RNN)
      ↓
Hidden States
      ↓
Attention
      ↓
Decoder (RNN)
      ↓
Output Sentence
```

---

## 6. Encoder

Encoder는 입력 문장을 순차적으로 처리하며
각 시점의 hidden state를 생성한다.

```python
h_t = RNN(x_t, h_{t-1})
```

역할:

* 문장의 의미를 벡터로 변환
* 모든 시점의 hidden state를 저장

---

## 7. Attention Mechanism (핵심)

Attention은 디코더가 출력 단어를 생성할 때
입력 문장의 어떤 부분에 집중할지 결정한다.

### 과정

1. Decoder hidden state와 Encoder hidden state 비교
2. 중요도(weight) 계산
3. weighted sum 수행

```python
context = Σ (attention_weight × encoder_hidden)
```

---

### 효과

* 긴 문장에서 성능 향상
* 중요한 단어에 집중 가능
* 정보 손실 감소

---

## 8. Decoder

Decoder는 이전 단어와 context vector를 입력으로 받아
다음 단어를 예측한다.

```python
y_t = Decoder(y_{t-1}, context)
```

---

## 9. Training

### Loss Function

```python
CrossEntropyLoss
```

→ 각 시점에서 단어 예측 정확도 측정

---

### Optimizer

```python
Adam
```

---

### 학습 과정

```
Input → Encoder → Attention → Decoder → Loss 계산 → Backpropagation
```

---

## 10. Execution

### 실행 방법

```bash
python train.py
```

---

### 데이터 경로

```python
DATA_PATH = "./data-2/eng-fra.txt"
```

---

## 11. Result

학습이 진행되면 Loss가 감소하는 것을 확인할 수 있다.

```
Epoch 1, Loss: 3.xxx
Epoch 2, Loss: 2.xxx
Epoch 3, Loss: 1.xxx
```

이는 모델이 번역 패턴을 학습하고 있음을 의미한다.

---

## 12. Key Insight

본 프로젝트의 핵심은 다음과 같다.

* Seq2Seq 구조의 한계 → **Attention으로 해결**
* 단순 RNN이 아닌 **context-aware 모델**
* 입력 문장의 모든 정보를 활용하여 출력 생성

---

## 13. Conclusion

Attention 메커니즘을 통해
기존 Seq2Seq 모델의 성능을 개선할 수 있음을 확인하였다.

특히 긴 문장이나 복잡한 문장에서
더 정확한 번역이 가능해진다.

---

