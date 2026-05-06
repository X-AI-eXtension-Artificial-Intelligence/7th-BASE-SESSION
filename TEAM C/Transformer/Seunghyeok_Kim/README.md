# ATTENTION 구현하기

## Model Architecture 공부

---

## load_data.py

- 데이터 로더 구현
- 데이터셋 로드 및 전처리
- 학습에 사용할 입력/정답 데이터 구성

---

## models

### blocks

#### encoder_layer.py

- Encoder Layer 로직 구현
- Self-Attention 적용
- Feed Forward Network 적용
- Residual Connection + Layer Normalization 구조 구현

#### decoder_layer.py

- Decoder Layer 로직 구현
- Masked Self-Attention 적용
- Encoder-Decoder Attention 적용
- Feed Forward Network 적용
- Residual Connection + Layer Normalization 구조 구현

---

### embedding

#### token_embedding.py

- 입력 token을 embedding vector로 변환
- 단어의 의미 정보를 벡터 형태로 표현

#### positional_encoding.py

- token의 위치 정보 생성
- Transformer가 문장 내 순서를 인식할 수 있도록 위치 정보 추가

#### transformer_embedding.py

- Token Embedding과 Positional Encoding 결합
- Transformer 입력 embedding 생성

---

### layers

#### scale_dot_product_attention.py

- Query, Key, Value를 이용한 Attention Score 계산
- Scaled Dot-Product Attention 구현
- Softmax를 통해 Attention Weight 계산
- Attention Weight와 Value를 곱해 Context Vector 생성

#### multi_head_attention.py

- Multi-Head Attention 구현
- 여러 개의 Head로 Attention 병렬 수행
- 각 Head의 결과를 Concatenate
- Linear Layer를 통해 최종 Attention Output 생성

#### position_wise_feed_forward.py

- Position-wise Feed Forward Network 구현
- 각 token 위치마다 동일한 MLP 적용
- Attention 이후 표현력 보강

#### layer_norm.py

- Layer Normalization 구현
- 학습 안정화
- Residual Connection 이후 정규화 수행

---

### model

#### encoder.py

- Encoder 전체 구조 구현
- 여러 개의 Encoder Layer를 Stack
- Source 문장을 Contextual Representation으로 변환

#### decoder.py

- Decoder 전체 구조 구현
- 여러 개의 Decoder Layer를 Stack
- Target 문장을 기반으로 다음 Token 예측
- Encoder Output을 참고하여 Decoding 수행

#### transformer.py

- Encoder와 Decoder를 결합한 전체 Transformer 구현
- Source Mask / Target Mask 생성
- 최종 Output Projection 수행
- Sequence-to-Sequence 구조 완성

---

## util

#### data_loader.py

- PyTorch DataLoader 구현
- Batch 단위 데이터 구성
- Padding 처리

#### tokenizer.py

- 문장 Tokenization 구현
- Vocabulary 생성
- Token ↔ Index 변환
- 특수 토큰 처리

#### bleu.py

- BLEU Score 계산
- 예측 문장과 정답 문장 비교
- 번역 품질 평가

#### epoch_timer.py

- Epoch별 학습 시간 측정
- 학습 진행 시간 기록

---

## train.py

- 학습 로직 구현
- Model Forward 수행
- Loss 계산
- Backpropagation 수행
- Optimizer Step 적용
- Validation 수행
- BLEU Score 기반 성능 평가

---

## conf.py

- 학습 설정값 관리
- Batch Size 설정
- Learning Rate 설정
- Epoch 수 설정
- Embedding Dimension 설정
- Head 수 및 Layer 수 설정
- Dropout 설정
- Device 설정

---

## graph.py

- 학습 결과 시각화
- Train / Validation Loss 그래프 출력
- 학습 과정 확인

---

## README.md

- 프로젝트 개요 작성
- 실행 방법 정리
- 폴더 구조 설명
- Transformer 구현 내용 요약
