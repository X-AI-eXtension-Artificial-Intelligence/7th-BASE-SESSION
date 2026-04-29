# Bahdanau Attention 구현 리뷰

논문 **"Neural Machine Translation by Jointly Learning to Align and Translate"**  
(Bahdanau et al., 2015) 의 Attention 메커니즘을 PyTorch로 구현한 리뷰 코드입니다.

---

## 논문 핵심 아이디어

기존 Seq2Seq는 소스 문장 전체를 고정 길이 벡터 하나로 압축해 Decoder에 넘깁니다.  
문장이 길어질수록 정보 손실이 발생하는 병목 문제가 있습니다.

Attention은 이를 해결하기 위해 Decoder가 매 타임스텝마다  
소스 전체를 다시 참조해 관련 있는 부분에 집중하도록 합니다.
Seq2Seq:   Encoder → [벡터 하나] → Decoder
Attention: Encoder → [annotation 시퀀스] → Attention → Decoder
↑
매 스텝마다 동적 참조

---

## 모델 구조
Encoder (BiGRU)          ← 논문 Section 3.2
↓ annotation 시퀀스
Attention (Bahdanau)     ← 논문 Section 3.1, Eq.(5)(6)
↓ context vector
Decoder (GRU)            ← 논문 Section 3.1, Eq.(4)

### Attention 수식
e_ij  = v_a^T * tanh(W_a * s_{i-1} + U_a * h_j)   # alignment score
α_ij  = softmax(e_ij)                               # attention weight
c_i   = Σ α_ij * h_j                                # context vector

---

## Toy Task

**숫자 시퀀스 정렬**
입력: [3, 1, 4, 1, 5]
출력: [1, 1, 3, 4, 5]

실제 번역 데이터 없이 바로 실행 가능하며,  
Attention이 잘 학습되면 히트맵에서 출력 단어 생성 시  
소스의 어느 위치를 참조했는지 확인할 수 있습니다.

---

## 파일 구조
attention-review/
├── README.md
├── requirements.txt
├── config.py               # 하이퍼파라미터
├── train.py                # 학습
├── evaluate.py             # 평가 + 시각화
│
├── model/
│   ├── init.py
│   ├── encoder.py          # BiGRU Encoder
│   ├── attention.py        # Bahdanau Attention
│   └── decoder.py          # Decoder
│
├── data/
│   ├── init.py
│   └── dataset.py          # 숫자 정렬 데이터 생성
│
└── utils/
├── init.py
└── visualize.py        # Attention weight 시각화

---

## 실행 환경

- Python 3.8+
- PyTorch 2.0+ (Meta AI)
- GPU 사용 시 CUDA 지원 환경 권장 (config.py에서 device 설정)

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