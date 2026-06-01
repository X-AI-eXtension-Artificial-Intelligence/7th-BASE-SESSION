# Vision Transformer (ViT) from Scratch

PyTorch로 구현한 Vision Transformer (ViT).
논문 [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020) 기반.

## Paper Overview

기존 컴퓨터 비전은 CNN이 주류였지만, ViT는 NLP에서 성공한 Transformer 구조를 이미지에 그대로 적용한 시도입니다.
핵심 아이디어는 이미지를 일정 크기의 패치로 나눠 각 패치를 NLP의 토큰처럼 취급하는 것입니다. 이렇게 만든 패치 시퀀스를 표준 Transformer Encoder에 그대로 입력합니다.

**주요 구성 요소:**

- **Patch Embedding** — 이미지를 `P×P` 크기의 패치로 분할하고 linear projection으로 벡터화
- **[CLS] Token** — 시퀀스 앞에 붙는 학습 가능한 토큰. 최종 분류에 사용 (BERT와 동일한 방식)
- **Position Embedding** — 패치 순서 정보를 주입하기 위한 학습 가능한 1D position embedding
- **Transformer Encoder** — Multi-Head Self-Attention + MLP 블록을 L번 반복. Pre-LN 구조 사용
- **Classification Head** — [CLS] 토큰의 출력을 Linear layer에 통과시켜 분류

**논문의 주요 발견:**

- 중간 규모 데이터셋(ImageNet 등)에서는 CNN보다 성능이 낮지만, 대규모 데이터(JFT-300M 등)로 사전학습하면 CNN을 능가
- Inductive bias(지역성, 이동 불변성)가 없는 대신, 충분한 데이터가 주어지면 전역적인 관계를 더 잘 학습

## Implementation

이 구현은 논문의 구조를 최대한 단순하게 재현하는 것을 목표로 합니다. 속도 최적화보다 가독성에 중점을 두었습니다.

```
vit.py      # 모델 구조 (PatchEmbeddings, MultiHeadAttention, Encoder 등)
train.py    # 학습 루프 및 config
data.py     # CIFAR-10 데이터 로딩 및 전처리
utils.py    # 체크포인트 저장/로드, attention 시각화
```

## Usage

**설치**

```bash
pip install -r requirements.txt
```

**학습**

```bash
python train.py --exp-name vit-cifar10 --epochs 100 --batch-size 256 --lr 1e-2
```

결과는 `experiments/<exp-name>/` 에 저장됩니다.

## Config

`train.py`에서 모델 구조를 수정할 수 있습니다.

```python
config = {
    "patch_size": 4,           # 32x32 이미지 → 8x8 = 64 패치
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}
```

> 논문의 ViT-Base는 12 layers, hidden size 768로 훨씬 크지만, 여기서는 구조 이해를 위해 경량 모델을 사용합니다.

## Results

CIFAR-10 기준 100 epoch 학습 결과 테스트 정확도 **75.5%** 달성.

## References

- [An Image is Worth 16x16 Words (arXiv)](https://arxiv.org/abs/2010.11929)
- [Original implementation](https://github.com/tintn/vision-transformer-from-scratch)