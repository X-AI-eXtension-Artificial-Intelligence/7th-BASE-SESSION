"""
train.py

Seq2Seq + Bahdanau Attention 기반 번역 모델 학습 실행 파일

이 파일의 핵심 목적

load_data.py에서 만든 DataLoader와 model.py에서 정의한 EncoderDecoder 모델을 연결하여
실제 학습, loss 계산, 역전파, 파라미터 업데이트, 간단한 greedy decoding 결과 출력을 수행.

전체 처리 흐름

1. 필요한 라이브러리와 사용자 정의 모듈 import
   - torch, torch.nn, optim, F
   - model.py: EncoderDecoder 모델 정의 포함
   - load_data.py: 데이터 로딩 및 DataLoader 생성 함수 포함

2. device 설정
   - GPU 사용 가능 시 cuda 사용
   - 불가능하면 cpu 사용
   - 모델과 tensor가 같은 device에 있어야 연산 가능

3. 주요 하이퍼파라미터 설정
   - PAD_idx: padding token ID
   - SOS_token: 문장 시작 token ID
   - EOS_token: 문장 종료 token ID
   - hidden_size: GRU hidden dimension 및 embedding dimension
   - batch_size: 한 batch에 들어가는 문장쌍 개수

4. train 함수
   - optimizer 생성
   - NLLLoss 생성
   - epoch 반복
   - batch 반복
   - train_step 호출
   - epoch별 평균 loss 출력

5. train_step 함수
   - optimizer gradient 초기화
   - model forward 수행
   - decoder output과 target을 loss 계산 가능 형태로 reshape
   - loss backward
   - optimizer step
   - loss 값 반환

6. ids2words 함수
   - 모델 입력/출력 ID sequence를 사람이 읽을 수 있는 단어 sequence로 복원

7. greedy_decode 함수
   - 학습된 모델로 한 batch를 추론
   - 각 time step에서 가장 확률이 높은 token 선택
   - 입력 문장, 정답 문장, 예측 문장 출력

8. main 실행부
   - DataLoader 생성
   - EncoderDecoder 모델 생성
   - 학습 실행
   - greedy decoding 결과 확인

주의점

- PAD_idx = 0, SOS_token = 0
  → padding과 SOS 토큰 ID 충돌
  → 실전 구현에서는 PAD=0, SOS=1, EOS=2처럼 분리 권장

- train 함수의 loss 평균 계산에서 loss / iter 사용
  → enumerate는 0부터 시작하므로 마지막 iter는 batch 개수 - 1
  → 정확한 평균은 loss / (iter + 1)

- target_tensor에는 EOS 토큰이 포함되지 않을 가능성
  → load_data.get_dataloader가 indexesFromSentence만 사용하기 때문
  → decoder가 문장 종료 시점을 명시적으로 학습하기 어려움

- greedy_decode는 EOS 기준 조기 종료 없음
  → 항상 max_len만큼 예측한 결과 출력

- ids2words는 padding ID 0도 단어로 변환
  → 현재 index2word[0] = "SOS"라서 padding 위치가 "SOS"로 출력될 가능성
  → PAD와 SOS가 같은 ID라 생기는 구조적 문제
"""


# PyTorch 메인 모듈 import
# tensor 연산, device 설정, no_grad context 사용을 위해 필요
import torch

# PyTorch neural network 모듈 import
# NLLLoss 같은 loss function 사용을 위해 필요
import torch.nn as nn

# PyTorch optimizer 모듈 import
# Adam optimizer 생성을 위해 필요
from torch import optim

# PyTorch functional API import
# 현재 train.py 안에서는 직접 사용하지 않음
# 이전 코드 스타일 또는 확장 가능성 때문에 남아 있는 import
import torch.nn.functional as F

# 사용자 정의 model.py import
# EncoderDecoder 모델 클래스를 사용하기 위해 필요
import model

# 사용자 정의 load_data.py import
# get_dataloader 함수로 학습 데이터를 만들기 위해 필요
import load_data


# GPU 사용 가능 여부에 따라 device 설정
# 모델과 입력 tensor를 같은 장치에 올려야 연산 가능
# cuda 사용 가능 시 학습 속도 향상 가능
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# padding token ID 설정
# loss 계산 시 padding 위치를 무시하기 위해 사용
# 현재 load_data.py에서 padding 값이 0이므로 PAD_idx도 0으로 설정
PAD_idx = 0

# 문장 시작 token ID 설정
# decoder가 첫 단어 생성을 시작할 때 사용하는 token
# 단, 현재 PAD_idx와 같은 0이라 의미 충돌 존재
SOS_token = 0

# 문장 종료 token ID 설정
# 원래는 decoder가 문장 끝을 학습하게 하기 위해 필요
# 현재 get_dataloader 구조에서는 실제 target에 EOS가 안 들어갈 가능성
EOS_token = 1

# hidden dimension 설정
# Encoder GRU, Decoder GRU, embedding layer의 차원으로 사용
# 값이 클수록 표현력은 증가하지만 연산량과 메모리 사용량 증가
hidden_size = 256

# batch size 설정
# 한 번의 optimizer update에 사용할 문장쌍 개수
# 값이 크면 gradient 추정이 안정적일 수 있으나 GPU 메모리 사용량 증가
batch_size = 32


# 전체 학습 루프 함수
# DataLoader를 받아 n_epochs 동안 모델을 학습시키기 위한 함수
def train(train_dataloader, model, n_epochs, learning_rate=0.0003):

    # Adam optimizer 생성
    # model.parameters()는 학습 가능한 모든 파라미터 반환
    # learning_rate는 파라미터 업데이트 크기 조절
    #
    # 필요한 이유:
    # - loss를 줄이는 방향으로 embedding, GRU, attention, linear layer 파라미터 업데이트
    # - Adam은 gradient의 1차/2차 모멘트를 이용해 학습률을 적응적으로 조정
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Negative Log Likelihood Loss 생성
    # model.py의 decoder는 log_softmax 결과를 반환하므로 NLLLoss와 짝이 맞음
    # ignore_index=PAD_idx는 target이 padding인 위치를 loss 계산에서 제외
    #
    # 필요한 이유:
    # - padding은 실제 정답 단어가 아니므로 loss에 포함되면 안 됨
    # - 번역 모델은 각 위치에서 정답 단어의 log-probability를 높이도록 학습
    criterion = nn.NLLLoss(ignore_index=PAD_idx)

    # epoch 반복
    # 전체 데이터셋을 n_epochs번 반복 학습
    for epoch in range(1, n_epochs + 1):

        # 현재 epoch의 누적 loss 초기화
        # batch별 loss를 더한 뒤 epoch 평균 loss 출력에 사용
        loss = 0

        # train_dataloader에서 batch 단위로 데이터 추출
        # iter는 batch index, batch는 TensorDataset에서 나온 tensor 묶음
        for iter, batch in enumerate(train_dataloader):

            # Batch tensors: [B, SeqLen]
            # batch 안의 tensor들은 load_data.py의 TensorDataset 순서와 동일

            # 입력 문장 token ID tensor
            # shape: [B, SeqLen]
            # Encoder 입력으로 사용
            input_tensor  = batch[0]

            # 입력 문장 mask tensor
            # shape: [B, SeqLen]
            # attention에서 padding 위치를 무시하기 위해 사용
            input_mask    = batch[1]

            # 출력 정답 문장 token ID tensor
            # shape: [B, SeqLen]
            # decoder teacher forcing 입력 및 loss target으로 사용
            target_tensor = batch[2]

            # 한 batch에 대한 학습 step 수행 후 loss 누적
            # train_step 내부에서 forward, loss 계산, backward, optimizer step 수행
            loss += train_step(input_tensor, input_mask, target_tensor,
                               model, optimizer, criterion)

        # epoch별 평균 loss 출력
        # 주의: iter는 0부터 시작하므로 엄밀히는 loss / (iter + 1)이 맞음
        # 현재 코드는 batch 수보다 1 작은 값으로 나누므로 loss가 약간 크게 출력됨
        print('Epoch {} Loss {}'.format(epoch, loss / iter))


# 한 batch에 대한 학습 step 함수
# forward → loss 계산 → backward → parameter update 수행
def train_step(input_tensor, input_mask, target_tensor, model,
               optimizer, criterion):

    # 이전 batch에서 계산된 gradient 초기화
    #
    # 필요한 이유:
    # - PyTorch는 기본적으로 gradient를 누적
    # - 매 batch마다 새 gradient만 사용하려면 zero_grad 필요
    optimizer.zero_grad()

    # 모델 forward 수행
    # target_tensor를 함께 넘기므로 decoder 내부에서 teacher forcing 사용
    #
    # decoder_outputs shape: [B, Seq, OutVocab]
    # decoder_hidden shape: [1, B, D]
    #
    # 필요한 이유:
    # - 현재 입력 문장에 대해 각 출력 위치별 단어 log-probability 계산
    decoder_outputs, decoder_hidden = model(input_tensor, input_mask, target_tensor)

    # Collapse [B, Seq] dimensions for NLL Loss
    # NLLLoss는 input shape [N, C], target shape [N] 형태를 기대
    # 따라서 batch 차원과 sequence 차원을 하나로 합침

    # loss 계산
    # decoder_outputs.view(-1, decoder_outputs.size(-1)):
    #   [B, Seq, OutVoc] → [B*Seq, OutVoc]
    #
    # target_tensor.view(-1):
    #   [B, Seq] → [B*Seq]
    #
    # 필요한 이유:
    # - 각 batch의 각 time step을 하나의 classification sample처럼 취급
    # - OutVoc 개 단어 중 정답 token 하나를 맞히는 다중분류 문제로 loss 계산
    loss = criterion(
        decoder_outputs.view(-1, decoder_outputs.size(-1)), # [B, Seq, OutVoc] -> [B*Seq, OutVoc]
        target_tensor.view(-1) # [B, Seq] -> [B*Seq]
    )

    # 역전파 수행
    # loss를 기준으로 모든 학습 가능한 파라미터의 gradient 계산
    #
    # 필요한 이유:
    # - embedding, GRU, attention, linear layer가 loss 감소 방향으로 업데이트되기 위함
    loss.backward()

    # optimizer를 통해 파라미터 업데이트
    # Adam 규칙에 따라 gradient 기반 weight update 수행
    optimizer.step()

    # Python float 형태의 loss 값 반환
    # epoch loss 누적과 출력에 사용
    return loss.item()


# 정수 ID sequence를 단어 sequence로 변환하는 함수
# 모델 입력/출력 결과를 사람이 읽을 수 있게 복원하기 위해 필요
def ids2words(lang, ids):

    # 각 token ID를 lang.index2word 사전을 사용해 단어로 변환
    # 예: [2, 5, 9] → ["i", "am", "cold"]
    #
    # 주의:
    # - padding ID 0도 "SOS"로 변환될 수 있음
    # - 현재 PAD와 SOS가 같은 ID이기 때문
    return [lang.index2word[idx] for idx in ids]


# 학습된 모델로 한 batch를 greedy decoding하는 함수
# 학습 결과를 간단히 눈으로 확인하기 위해 필요
def greedy_decode(model, dataloader, input_lang, output_lang):

    # gradient 계산 비활성화
    #
    # 필요한 이유:
    # - 추론 단계에서는 parameter update가 없음
    # - gradient 저장을 하지 않아 메모리 사용량 감소
    # - 연산 속도 측면에서도 유리
    with torch.no_grad():

        # dataloader에서 첫 번째 batch 하나만 가져옴
        # 전체 평가가 아니라 샘플 출력 확인용
        batch = next(iter(dataloader))

        # 입력 문장 token ID tensor
        # shape: [B, SeqLen]
        input_tensor  = batch[0]

        # 입력 문장 mask tensor
        # attention에서 padding 위치를 무시하기 위해 사용
        input_mask    = batch[1]

        # 정답 출력 문장 token ID tensor
        # 예측 결과와 비교 출력하기 위해 사용
        target_tensor = batch[2]

        # target_tensor 없이 모델 forward 수행
        # target이 없으므로 decoder는 teacher forcing이 아니라 greedy 방식으로 작동
        #
        # decoder_outputs shape: [B, Seq, OutVocab]
        decoder_outputs, decoder_hidden = model(input_tensor, input_mask)

        # 각 time step에서 log-probability가 가장 높은 token 선택
        # topi shape은 대략 [B, Seq, 1]
        topv, topi = decoder_outputs.topk(1)

        # 마지막 차원 제거
        # decoded_ids shape: [B, Seq]
        # 각 sample의 예측 token ID sequence
        decoded_ids = topi.squeeze()

        # batch 안의 각 sample 순회
        for idx in range(input_tensor.size(0)):

            # 입력 token ID sequence를 입력 언어 단어 sequence로 변환
            # 프랑스어 입력 문장 확인 목적
            input_sent = ids2words(input_lang, input_tensor[idx].cpu().numpy())

            # 예측 token ID sequence를 출력 언어 단어 sequence로 변환
            # 모델 번역 결과 확인 목적
            output_sent = ids2words(output_lang, decoded_ids[idx].cpu().numpy())

            # 정답 target token ID sequence를 출력 언어 단어 sequence로 변환
            # 예측 결과와 정답 비교 목적
            target_sent = ids2words(output_lang, target_tensor[idx].cpu().numpy())

            # 입력 문장 출력
            print('Input:  {}'.format(input_sent))

            # 정답 문장 출력
            print('Target: {}'.format(target_sent))

            # 모델 예측 문장 출력
            print('Output: {}'.format(output_sent))


# 이 파일을 직접 실행할 때만 아래 코드 실행
# 다른 파일에서 import할 때는 실행되지 않음
if __name__ == '__main__':

    # 학습 데이터 준비
    #
    # load_data.get_dataloader 내부에서 수행되는 일:
    # - data/eng-fra.txt 읽기
    # - 문장 정규화
    # - 문장쌍 필터링
    # - 단어 사전 생성
    # - input_ids, input_mask, target_ids, target_mask 생성
    # - DataLoader 생성
    #
    # 반환값:
    # - input_lang: 입력 언어 단어 사전
    # - output_lang: 출력 언어 단어 사전
    # - train_dataloader: 학습 batch 공급 객체
    input_lang, output_lang, train_dataloader = load_data.get_dataloader(batch_size)

    # EncoderDecoder 모델 생성
    #
    # hidden_size:
    # - embedding dimension
    # - encoder GRU hidden dimension
    # - decoder GRU hidden dimension
    #
    # input_lang.n_words:
    # - encoder embedding의 vocabulary size
    #
    # output_lang.n_words:
    # - decoder embedding과 output linear layer의 vocabulary size
    #
    # .to(device):
    # - 모델 파라미터를 CPU 또는 GPU로 이동
    model = model.EncoderDecoder(hidden_size, input_lang.n_words, output_lang.n_words).to(device)

    # 모델 학습 실행
    # 전체 train_dataloader를 20 epoch 반복
    train(train_dataloader, model, n_epochs=20)

    # 학습 후 한 batch에 대해 greedy decoding 결과 출력
    # 학습이 실제로 어느 정도 되었는지 빠르게 확인하는 용도
    greedy_decode(model, train_dataloader, input_lang, output_lang)