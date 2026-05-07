# 코드 이해 핵심 노트

## 1. 전체 흐름

이 저장소는 논문 **Attention Is All You Need**의 Transformer를 PyTorch로 직접 구현한 예제입니다.

전체 학습 흐름은 다음과 같습니다.

```text
Multi30k 독일어-영어 병렬 말뭉치
↓
spaCy 토크나이징
↓
torchtext Field / BucketIterator
↓
Encoder 입력: source 문장
Decoder 입력: target 문장을 한 칸 오른쪽으로 민 것
↓
Transformer forward
↓
CrossEntropyLoss
↓
BLEU 평가
```

## 2. Encoder와 Decoder의 차이

- Encoder는 source 문장을 읽고, 문맥 벡터를 만듭니다.
- Decoder는 이전 target 토큰과 Encoder 출력값을 함께 보면서 다음 target 토큰을 예측합니다.

Decoder에는 Attention이 2번 있습니다.

1. **Masked Self-Attention**: 정답 문장의 미래 단어를 훔쳐보지 못하게 막음
2. **Encoder-Decoder Attention**: source 문장 정보를 참고함

## 3. Mask가 중요한 이유

Transformer는 RNN처럼 순서대로 읽는 구조가 아니라, 문장 전체를 한 번에 행렬 연산합니다. 그래서 두 가지 mask가 필요합니다.

- `src_mask`: padding 토큰을 attention에서 제외
- `trg_mask`: padding 제외 + 미래 토큰 가림

## 4. 주의할 점

- 원 코드는 2019년 기준으로 작성되어 `torchtext.legacy`에 의존합니다.
- 최신 torchtext에서는 `Field`, `BucketIterator`, `Multi30k.splits` 방식이 바뀌었거나 제거되었습니다.
- `train.py`의 loss에서 `ignore_index=src_pad_idx`를 사용하지만, 일반적으로 target 예측 loss에서는 `trg_pad_idx`를 쓰는 편이 더 자연스럽습니다. 다만 이 코드에서는 source/target vocab의 `<pad>` 인덱스가 보통 같아서 실제 문제가 드러나지 않을 수 있습니다.
- `Transformer` 클래스에 `trg_sos_idx`가 저장되지만 forward에서 직접 쓰이지 않습니다. 시작 토큰은 데이터 전처리 단계에서 target 문장 앞에 붙습니다.
