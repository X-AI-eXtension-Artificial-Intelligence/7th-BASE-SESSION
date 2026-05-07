# Bahdanau Attention 구현 리뷰

> 논문 **"Neural Machine Translation by Jointly Learning to Align and Translate"**  
> Bahdanau et al., 2015 의 Attention 메커니즘을 PyTorch로 구현한 리뷰 코드입니다.

---

## 논문 핵심 아이디어

기존 Seq2Seq는 소스 문장 전체를 **고정 길이 벡터 하나**로 압축해 Decoder에 넘깁니다.  
문장이 길어질수록 정보 손실이 발생하는 **병목 문제**가 있습니다.

Attention은 Decoder가 매 타임스텝마다 소스 전체를 다시 참조해  
관련 있는 부분에 **동적으로 집중**하도록 합니다.

```
Seq2Seq:   Encoder ──► [벡터 하나] ──► Decoder

Attention: Encoder ──► [h1, h2, h3, ..., hT]
                               │
                          Attention (매 스텝마다 동적 참조)
                               │
                            Decoder
```

---

## 모델 구조

```
Encoder (BiGRU)        ← 논문 Section 3.2
    │
    │  annotation 시퀀스 h_j
    ▼
Attention (Bahdanau)   ← 논문 Section 3.1, Eq.(5)(6)
    │
    │  context vector c_i
    ▼
Decoder (GRU)          ← 논문 Section 3.1, Eq.(4)
```

### Attention 수식

$$e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T} \exp(e_{ik})}$$

$$c_i = \sum_{j=1}^{T} \alpha_{ij} h_j$$

| 기호 | 설명 |
|------|------|
| $s_{i-1}$ | Decoder의 이전 hidden state |
| $h_j$ | Encoder의 j번째 annotation |
| $\alpha_{ij}$ | i번째 출력 생성 시 j번째 입력에 대한 attention weight |
| $c_i$ | i번째 context vector |

---

## Toy Task

**숫자 시퀀스 정렬**

```
입력: [3, 1, 4, 1, 5]
출력: [1, 1, 3, 4, 5]
```

실제 번역 데이터 없이 바로 실행 가능합니다.  
Attention이 잘 학습되면 히트맵에서 출력 단어 생성 시 소스의 어느 위치를 참조했는지 확인할 수 있습니다.

---

## 파일 구조

```
attention-review/
├── README.md
├── requirements.txt
├── config.py               # 하이퍼파라미터
├── train.py                # 학습
├── evaluate.py             # 평가 + 시각화
│
├── model/
│   ├── __init__.py
│   ├── encoder.py          # BiGRU Encoder
│   ├── attention.py        # Bahdanau Attention
│   └── decoder.py          # Decoder
│
├── data/
│   ├── __init__.py
│   └── dataset.py          # 숫자 정렬 데이터 생성
│
└── utils/
    ├── __init__.py
    └── visualize.py        # Attention weight 시각화
```

---

## 실행

```bash
# 패키지 설치
pip install -r requirements.txt

# 학습
python train.py

# 평가 + Attention 히트맵 저장
python evaluate.py
```

학습 완료 후 `attention_map_0.png`, `attention_map_1.png`, `attention_map_2.png` 가 생성됩니다.

---

## 참고 논문

Bahdanau, D., Cho, K., & Bengio, Y. (2015).  
Neural Machine Translation by Jointly Learning to Align and Translate.  
ICLR 2015. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
