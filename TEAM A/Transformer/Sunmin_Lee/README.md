# Transformer - "Attention Is All You Need" PyTorch 구현

> **원본 레포**: [hyunwoongko/transformer](https://github.com/hyunwoongko/transformer)  
> **논문**: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

---

## 개요

이 레포는 "Attention Is All You Need" 논문을 공부하면서, 원본 구현 코드를 분석하고 실험적으로 수정한 것입니다.

**원본 코드에서 추가/수정한 내용:**

- `conf.py`: 실험별 하이퍼파라미터 설정 분리 (base / small / large / custom)
- `util/attention_visualizer.py`: Multi-Head Attention 시각화 코드 추가
- `run_experiments.py`: 다중 실험 자동 실행 스크립트 추가
- 전체 코드에 한국어 상세 주석 추가 (논문 내용과 매핑)

---

## 디렉토리 구조

```
transformer/
│
├── models/                          # 모델 핵심 코드
│   ├── model/
│   │   ├── transformer.py           # 전체 Transformer 통합 모델
│   │   ├── encoder.py               # 인코더 (N개 레이어 스택)
│   │   └── decoder.py               # 디코더 (N개 레이어 스택)
│   │
│   ├── blocks/
│   │   ├── encoder_layer.py         # 인코더 단일 레이어
│   │   │                            # (Self-Attention + FFN)
│   │   └── decoder_layer.py         # 디코더 단일 레이어
│   │                                # (Masked Att + Enc-Dec Att + FFN)
│   ├── layers/
│   │   ├── scale_dot_product_attention.py  # Scaled Dot-Product Attention
│   │   ├── multi_head_attention.py         # Multi-Head Attention
│   │   ├── layer_norm.py                   # Layer Normalization
│   │   └── position_wise_feed_forward.py  # Position-wise FFN
│   │
│   └── embedding/
│       ├── transformer_embedding.py  # 토큰 + 위치 임베딩 통합
│       ├── positional_encoding.py    # Positional Encoding (sin/cos)
│       └── token_embeddings.py       # 토큰 임베딩
│
├── util/
│   ├── data_loader.py               # Multi30K 데이터 로딩
│   ├── tokenizer.py                 # 토크나이저
│   ├── bleu.py                      # BLEU 점수 계산
│   └── attention_visualizer.py      # ✅ 추가: Attention 시각화
│
├── saved/
│   ├── transformer-base/            # 학습된 모델 저장
│   ├── attention_maps/              # ✅ 추가: Attention 시각화 이미지
│   └── experiment_results/          # ✅ 추가: 실험별 로그
│
├── conf.py                          # ✅ 수정: 실험별 설정 분리
├── data.py                          # 데이터 전처리
├── train.py                         # 학습 실행
├── graph.py                         # 학습 결과 시각화
└── run_experiments.py               # ✅ 추가: 다중 실험 자동 실행
```

---

## 실험 설정

`conf.py`의 `experiment` 변수를 변경해서 다양한 설정을 실험할 수 있습니다.

| 설정 | d_model | n_layers | n_heads | ffn_hidden | 목적 |
|------|---------|----------|---------|------------|------|
| `base`   | 512 | 6 | 8  | 2048 | 논문 원본 설정 |
| `small`  | 256 | 3 | 4  | 1024 | 경량 실험용 |
| `large`  | 1024| 8 | 16 | 4096 | 대형 실험용 |
| `custom` | 512 | 6 | 1  | 2048 | Single-Head vs Multi-Head 비교 |

```python
# conf.py 에서 experiment 변수만 바꾸면 됩니다
experiment = "base"   # "small", "large", "custom" 중 선택
```

---

## 설치 및 실행

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 패키지 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torchtext spacy matplotlib

# 언어 모델 설치
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### 2. GPU 확인

```bash
python -c "import torch; print('GPU 사용 가능:', torch.cuda.is_available())"
```

### 3. 학습 실행

```bash
# 단일 실험 (conf.py의 experiment 설정 사용)
python train.py

# 다중 실험 자동 실행 (base / small / custom 순서로)
python run_experiments.py
```

### 4. 결과 시각화

```bash
# 학습 Loss 그래프
python graph.py
```

---

## 실험 결과

### 하이퍼파라미터 실험 (Multi30K EN-DE)

| 설정 | 파라미터 수 | Train Loss | Val Loss | BLEU |
|------|------------|------------|----------|------|
| base (논문 원본) | 55M | 2.85 | 3.20 | 26.4 |
| small (경량) | 약 14M | - | - | - |
| custom (Single-Head) | 55M | - | - | - |

> 실험 진행 중 업데이트 예정

### Multi-Head Attention 시각화

학습 후 `saved/attention_maps/` 폴더에서 각 레이어/헤드별 Attention 가중치 이미지를 확인할 수 있습니다.

- 헤드마다 서로 다른 언어적 관계를 학습하는 것을 시각적으로 확인
- 레이어가 깊어질수록 더 추상적인 관계에 집중하는 것을 확인

---

## 논문 핵심 내용 정리

### Transformer 전체 구조

```
입력 시퀀스
    ↓
[임베딩 + Positional Encoding]
    ↓
┌────────────────────────┐
│      인코더 × N        │
│  Multi-Head Self-Att   │  → 입력 문장 내 토큰 간 관계 학습
│  Feed-Forward          │  → 각 토큰 표현 강화
└────────────────────────┘
    ↓ K, V 전달
┌────────────────────────┐
│      디코더 × N        │
│  Masked Self-Att       │  → 미래 토큰 차단 (치팅 방지)
│  인코더-디코더 Att      │  → 입력 전체 참조
│  Feed-Forward          │
└────────────────────────┘
    ↓
[Linear + Softmax]
    ↓
출력 시퀀스
```

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Q (Query)**: 내가 찾는 것
- **K (Key)**: 각 토큰이 가진 태그
- **V (Value)**: 실제 전달할 정보
- **√d_k 로 나누는 이유**: 내적값이 커질수록 softmax가 극단화되어 gradient 소실 방지

### 왜 Self-Attention인가

| 방식 | 병렬화 | 장거리 의존성 경로 | 레이어당 복잡도 |
|------|--------|------------------|----------------|
| RNN  | ❌ | O(n) | O(n·d²) |
| CNN  | ✅ | O(log n) | O(k·n·d²) |
| **Self-Attention** | ✅ | **O(1)** | O(n²·d) |

---

## 참고

- [Attention is All You Need, 2017](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer - Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
- [원본 구현 - hyunwoongko/transformer](https://github.com/hyunwoongko/transformer)

---

## License

Apache License 2.0 (원본 라이센스 유지)