# hyunwoongko/transformer 코드 주석본

이 자료는 GitHub 저장소 `hyunwoongko/transformer`의 PyTorch Transformer 구현을 **공부용으로 이해하기 쉽게 한국어 주석을 추가한 버전**입니다.

## 구성

```text
annotated/
├── conf.py                         # 학습 하이퍼파라미터/경로 설정
├── data.py                         # 토크나이저·데이터셋·Iterator 생성
├── graph.py                        # train/validation loss 시각화
├── train.py                        # 학습 루프, 평가 루프, BLEU 계산
├── models/
│   ├── blocks/                     # EncoderLayer, DecoderLayer
│   ├── embedding/                  # Token/Position/Transformer embedding
│   ├── layers/                     # Attention, FFN, LayerNorm
│   └── model/                      # Encoder, Decoder, Transformer wrapper
└── util/                           # DataLoader, Tokenizer, BLEU, timer
```

## 먼저 알아둘 점

- 원 저장소는 `torchtext.legacy`, `spacy` 구버전 사용을 전제로 작성되어 있습니다.
- 최신 PyTorch / torchtext 환경에서는 그대로 실행되지 않을 수 있습니다.
- 이 주석본은 **학습 구조 이해**에 초점을 맞췄고, 실전 프로젝트라면 최신 torchtext 또는 Hugging Face Datasets 방식으로 데이터 파이프라인을 바꾸는 편이 좋습니다.

## 읽는 순서 추천

1. `conf.py` — 모델 크기와 학습 설정 확인
2. `data.py`, `util/data_loader.py`, `util/tokenizer.py` — 데이터가 어떻게 들어오는지 확인
3. `models/embedding/*` — 단어 임베딩과 위치 인코딩 확인
4. `models/layers/*` — Attention, FFN, LayerNorm 확인
5. `models/blocks/*` — EncoderLayer / DecoderLayer 확인
6. `models/model/transformer.py` — 전체 Transformer 연결 구조 확인
7. `train.py` — 실제 학습 흐름 확인

## 라이선스 메모

원 저장소는 Apache License 2.0을 사용합니다. 이 파일들은 원 코드의 구조를 학습용으로 주석 처리한 2차 자료이며, 원 저장소 및 라이선스 고지를 함께 확인해야 합니다.
