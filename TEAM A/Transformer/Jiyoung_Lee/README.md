#  Transformer 코드 분석 (hyunwoongko/transformer)

> **원본 레포**: [https://github.com/hyunwoongko/transformer](https://github.com/hyunwoongko/transformer)  
> **논문**: Attention Is All You Need (Vaswani et al., 2017, Google Brain)  
> **구현**: PyTorch 기반 Encoder-Decoder Transformer (번역 태스크: Multi30K)

---

## 프로젝트 구조

```
transformer/
├── conf.py                          # 하이퍼파라미터 설정
├── train.py                         # 학습 루프
├── data.py                          # 데이터 로딩 (Multi30K)
│
└── models/
    ├── embedding/
    │   ├── token_embeddings.py      # 토큰 → 벡터 변환
    │   ├── positional_encoding.py   # 위치 정보 인코딩 (sin/cos)
    │   └── transformer_embedding.py # 위 둘을 합친 최종 임베딩
    │
    ├── layers/
    │   ├── scale_dot_product_attention.py  #  Attention 핵심 연산
    │   ├── multi_head_attention.py         #  Multi-Head Attention
    │   ├── position_wise_feed_forward.py   # FFN (2-layer MLP)
    │   └── layer_norm.py                   # Layer Normalization
    │
    ├── blocks/
    │   ├── encoder_layer.py         # EncoderLayer 1개 블록
    │   └── decoder_layer.py         # DecoderLayer 1개 블록
    │
    └── model/
        ├── encoder.py               # EncoderLayer × N 스택
        ├── decoder.py               # DecoderLayer × N 스택
        └── transformer.py           # 최상위: Encoder + Decoder 조립
```

---

##  핵심 하이퍼파라미터 (`conf.py`)

| 파라미터 | 값 | 의미 |
|---|---|---|
| `d_model` | 512 | 모든 레이어의 벡터 차원 |
| `n_layers` | 6 | Encoder/Decoder 반복 횟수 |
| `n_heads` | 8 | Multi-Head Attention의 헤드 수 |
| `ffn_hidden` | 2048 | FFN 중간 레이어 차원 (= d_model × 4) |
| `drop_prob` | 0.1 | Dropout 확률 |
| `max_len` | 256 | 최대 시퀀스 길이 |
| `batch_size` | 128 | 배치 크기 |

---

##  데이터 흐름 (전체 forward pass)

```
[입력 문장 토큰 IDs]  →  TransformerEmbedding  →  EncoderLayer × 6  →  enc_src
                                                                              ↓
[출력 문장 토큰 IDs]  →  TransformerEmbedding  →  DecoderLayer × 6 (+ cross-attention with enc_src)
                                                                              ↓
                                                                      Linear (LM Head)
                                                                              ↓
                                                                    Output logits (vocab size)
```

---

##  파일별 핵심 코드 요약

### 1. `positional_encoding.py` — 위치 정보 주입

토큰은 순서가 없는 집합이라 위치 정보를 별도로 넣어줘야 함. sin/cos 함수로 각 위치마다 고유한 패턴 생성.

```python
self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))  # 짝수 차원
self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))  # 홀수 차원
```

- `requires_grad = False` → 학습되지 않는 고정 값
- 출력 shape: `[seq_len, d_model]`

---

### 2. `scale_dot_product_attention.py` — Attention 핵심 연산

```python
def forward(self, q, k, v, mask=None, e=1e-12):
    k_t = k.transpose(2, 3)
    score = (q @ k_t) / math.sqrt(d_tensor)  # Q·Kᵀ / √d_k

    if mask is not None:
        score = score.masked_fill(mask == 0, -10000)  # 마스킹

    score = self.softmax(score)   # 확률로 변환
    v = score @ v                 # Value 가중합
    return v, score
```

**흐름 요약**: Q·Kᵀ → scale → (mask) → softmax → ·V

- **mask**: padding 토큰(-10000으로 채워 softmax 후 ≈0) 또는 미래 토큰 가리기 (decoder)

---

### 3. `multi_head_attention.py` — 다중 관점 Attention

```python
def forward(self, q, k, v, mask=None):
    q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)  # Linear 투영
    q, k, v = self.split(q), self.split(k), self.split(v)  # head로 분할
    out, attention = self.attention(q, k, v, mask=mask)    # Attention 계산
    out = self.concat(out)   # head 다시 합치기
    out = self.w_concat(out) # 최종 Linear
    return out
```

- `split()`: `[batch, seq, d_model]` → `[batch, head, seq, d_model/head]`
- `concat()`: `split()`의 역연산

---

### 4. `encoder_layer.py` — Encoder 1개 블록

```python
def forward(self, x, src_mask):
    _x = x
    x = self.attention(q=x, k=x, v=x, mask=src_mask)  # Self-Attention (Q=K=V=입력)
    x = self.norm1(x + _x)   # Residual connection + LayerNorm

    _x = x
    x = self.ffn(x)           # Feed Forward Network
    x = self.norm2(x + _x)   # Residual connection + LayerNorm
    return x
```

**구조**: Self-Attention → Add & Norm → FFN → Add & Norm

---

### 5. `decoder_layer.py` — Decoder 1개 블록 (3단 구성)

```python
def forward(self, dec, enc, trg_mask, src_mask):
    # 1. Masked Self-Attention (미래 토큰 못 봄)
    x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
    x = self.norm1(x + dec)

    # 2. Cross-Attention (Q=decoder, K=V=encoder 출력) ← 핵심!
    x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
    x = self.norm2(x + _x)

    # 3. FFN
    x = self.ffn(x)
    x = self.norm3(x + _x)
    return x
```

**Encoder와 차이점**: Self-Attention이 Masked + Cross-Attention 1개 추가

---

### 6. `transformer.py` — 마스크 생성 & 전체 조립

```python
def make_src_mask(self, src):
    # PAD 토큰 위치를 False로 → Attention에서 무시
    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask

def make_trg_mask(self, trg):
    trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
    trg_len = trg.shape[1]
    # 하삼각 행렬: 현재 위치 이후 미래 토큰은 모두 0 (못 봄)
    trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len))
    trg_mask = trg_pad_mask & trg_sub_mask  # 두 마스크 AND
    return trg_mask
```

---

## 🔗 참고 자료

- [Attention Is All You Need 논문](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [원본 레포 README](https://github.com/hyunwoongko/transformer)
