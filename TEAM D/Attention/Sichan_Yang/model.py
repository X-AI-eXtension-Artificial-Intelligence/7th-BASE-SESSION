"""
model.py

Seq2Seq 기반 프랑스어 → 영어 번역 모델 정의 파일

이 파일의 핵심 목적

문장 번역 모델의 신경망 구조 정의.
load_data.py가 문장을 정수 ID tensor로 바꾸면,
이 파일의 모델은 그 정수 ID sequence를 입력받아 target 언어의 단어 sequence를 예측.

전체 모델 구조

1. EncoderRNN
   - 입력 문장 token ID를 embedding vector로 변환
   - GRU를 통해 입력 문장의 순서 정보와 문맥 정보 인코딩
   - 각 입력 위치의 hidden state인 encoder_outputs 반환
   - 마지막 hidden state인 encoder_hidden 반환

2. DecoderRNN
   - attention 없는 기본 decoder
   - encoder_hidden 하나만 이용해 출력 문장을 생성
   - 현재 EncoderDecoder에서는 사용되지 않음
   - attention decoder와 비교하기 위한 기본 구조

3. BahdanauAttention
   - decoder가 출력 단어를 생성할 때 입력 문장의 어느 위치를 볼지 계산
   - additive attention 방식 사용
   - 입력 문장의 모든 encoder hidden state에 대해 attention score 계산
   - padding 위치는 mask로 제거
   - attention weight를 이용해 context vector 생성

4. AttnDecoder
   - attention을 사용하는 decoder
   - 매 decoding step마다 attention context 계산
   - 현재 생성 중인 단어 embedding과 context vector를 결합
   - GRU로 다음 hidden state와 출력 단어 분포 생성

5. EncoderDecoder
   - EncoderRNN과 AttnDecoder를 하나의 모델로 결합
   - train.py에서 실제로 호출되는 최종 모델 클래스

중요한 shape 흐름

입력:
    inputs: [B, Seq]
    input_mask: [B, Seq]

Encoder 출력:
    encoder_outputs: [B, Seq, D]
    encoder_hidden: [1, B, D]

Attention Decoder 출력:
    decoder_outputs: [B, max_len, output_vocab_size]
    decoder_hidden: [1, B, D]

기호:
    B: batch size
    Seq: 입력 문장 길이
    D: hidden size
    output_vocab_size: 출력 언어 단어장 크기

주의점

- DecoderRNN은 정의되어 있지만 현재 최종 모델에서는 사용되지 않음
  EncoderDecoder 내부에서 AttnDecoder 사용 중

- BahdanauAttention 내부의 W3는 선언되어 있으나 forward에서 사용되지 않음
  불필요한 layer

- scores.data.masked_fill_ 사용
  .data 사용은 autograd 추적을 우회할 수 있으므로 권장되지 않음
  scores = scores.masked_fill(...) 형태가 더 안전

- greedy decoding에서 decoder_output.data.topk(1) 사용
  .data 사용보다 decoder_output.topk(1) 또는 detach() 사용 권장

- encoder_hidden을 decoder_hidden으로 그대로 사용
  encoder와 decoder hidden size가 같기 때문에 가능
  구조가 달라지면 bridge layer 필요
"""


# PyTorch 메인 모듈 import
# tensor 생성, device 설정, 신경망 연산을 위해 필요
import torch

# PyTorch neural network 모듈 import
# nn.Module, Embedding, GRU, Linear 같은 layer 정의를 위해 필요
import torch.nn as nn

# PyTorch optimizer 모듈 import
# 현재 model.py 내부에서는 직접 사용하지 않음
# train.py에서 optimizer를 정의하므로 이 파일에서는 불필요한 import
from torch import optim

# PyTorch functional API import
# relu, log_softmax, softmax 같은 함수형 연산 사용을 위해 필요
import torch.nn.functional as F


# GPU 사용 가능 시 cuda, 아니면 cpu 사용
# 모델 내부에서 새 tensor를 만들 때 같은 device에 올리기 위해 필요
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Encoder RNN 클래스
# 입력 문장을 hidden state sequence로 인코딩하는 역할
# Seq2Seq 구조에서 source sentence를 vector representation으로 바꾸기 위해 필요
class EncoderRNN(nn.Module):

    # Encoder 초기화
    # input_size: 입력 언어 단어장 크기
    # hidden_size: embedding 차원 및 GRU hidden 차원
    def __init__(self, input_size, hidden_size):

        # 부모 클래스 nn.Module 초기화
        # PyTorch 모델이 parameter 추적, device 이동, train/eval mode 전환 등을 할 수 있게 하기 위해 필요
        super(EncoderRNN, self).__init__()

        # hidden size 저장
        # embedding 차원과 GRU hidden 차원 설정에 사용
        self.hidden_size = hidden_size

        # 입력 단어 ID를 dense vector로 변환하는 embedding layer
        # 신경망은 discrete token ID 자체보다 연속 벡터 표현을 학습해야 하므로 필요
        # 입력 shape: [B, Seq]
        # 출력 shape: [B, Seq, hidden_size]
        self.embedding = nn.Embedding(input_size, hidden_size)

        # 입력 embedding sequence를 순차적으로 처리하는 GRU layer
        # 문장 내 단어 순서와 앞뒤 문맥 정보를 hidden state에 누적하기 위해 필요
        # batch_first=True이므로 입력 shape은 [B, Seq, D]
        # 출력 output shape: [B, Seq, hidden_size]
        # 출력 hidden shape: [1, B, hidden_size]
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    # Encoder forward 연산
    # input: 입력 문장 token ID tensor, shape [B, Seq]
    def forward(self, input):

        # 입력 token ID를 embedding vector sequence로 변환
        # GRU는 정수 ID가 아니라 실수 벡터 sequence를 입력으로 받기 때문에 필요
        embedded = self.embedding(input)

        # embedding sequence를 GRU에 통과
        # output은 모든 time step의 hidden state
        # hidden은 마지막 time step의 hidden state
        # attention decoder는 output 전체를 사용하고, decoder 초기 hidden에는 hidden 사용
        output, hidden = self.gru(embedded)

        # encoder output sequence와 마지막 hidden state 반환
        # decoder가 문장 생성을 시작하고 attention 계산을 수행하기 위해 필요
        return output, hidden


# Attention 없는 기본 Decoder RNN 클래스
# encoder_hidden 하나만 보고 출력 문장을 생성하는 기본 Seq2Seq decoder
# 현재 최종 모델에서는 사용되지 않지만 attention decoder와 구조 비교 가능
class DecoderRNN(nn.Module):

    # Standard non-attentional decoder
    # 기본적인 non-attention decoder라는 의미
    def __init__(self, hidden_size, output_size):

        # 부모 클래스 nn.Module 초기화
        # PyTorch layer와 parameter 등록을 위해 필요
        super(DecoderRNN, self).__init__()

        # 출력 언어의 token ID를 embedding vector로 변환하는 layer
        # decoder 입력은 이전에 생성한 단어 ID이므로 이를 vector로 바꾸기 위해 필요
        self.embedding = nn.Embedding(output_size, hidden_size)

        # decoder GRU
        # 이전 단어 embedding과 이전 hidden state를 이용해 다음 hidden state 생성
        # attention이 없으므로 입력 크기는 hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # GRU hidden state를 출력 언어 단어장 크기의 score로 변환하는 linear layer
        # 각 단어가 다음 단어일 점수를 계산하기 위해 필요
        self.out = nn.Linear(hidden_size, output_size)

    # Decoder 전체 forward
    # encoder_outputs: attention decoder와 interface를 맞추기 위해 받지만, 이 기본 decoder에서는 사용하지 않음
    # encoder_hidden: encoder 마지막 hidden state, decoder 초기 hidden으로 사용
    # input_mask: attention 없는 decoder에서는 사용하지 않음
    # target_tensor: 학습 시 teacher forcing에 사용
    # SOS_token: 첫 decoder 입력 token
    # max_len: 생성할 최대 출력 길이
    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):

        # Teacher forcing if given a target_tensor, otherwise greedy.
        # target_tensor가 있으면 정답 token을 다음 입력으로 사용
        # target_tensor가 없으면 모델이 예측한 token을 다음 입력으로 사용

        # batch size 추출
        # decoder 시작 입력을 batch 크기만큼 만들기 위해 필요
        batch_size = encoder_outputs.size(0)

        # 첫 decoder 입력 생성
        # 모든 batch sample의 첫 입력을 SOS_token으로 채움
        # decoder는 SOS를 보고 첫 번째 출력 단어를 예측해야 하므로 필요
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)

        # decoder 초기 hidden state 설정
        # encoder가 입력 문장을 요약한 마지막 hidden state를 decoder 시작 상태로 사용
        # encoder와 decoder hidden_size가 같기 때문에 직접 전달 가능
        decoder_hidden = encoder_hidden # TODO: Consider bridge

        # 각 time step의 decoder output 저장 리스트
        # 마지막에 sequence 전체 출력으로 concat하기 위해 필요
        decoder_outputs = []

        # max_len만큼 출력 단어 생성 반복
        # 현재 코드는 EOS 기반 조기 종료 없이 고정 길이 생성
        for i in range(max_len):

            # 현재 decoder_input과 decoder_hidden으로 한 step 예측
            # decoder_output shape: [B, 1, output_vocab_size]
            # decoder_hidden shape: [1, B, hidden_size]
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)

            # 현재 step의 출력 score 저장
            # 전체 출력 sequence를 만들기 위해 필요
            decoder_outputs.append(decoder_output)

            # target_tensor가 주어진 경우
            # 학습 단계의 teacher forcing 수행
            if target_tensor is not None:

                # 정답 target의 i번째 token을 다음 decoder 입력으로 사용
                # 모델이 이전 step에서 틀려도 다음 step 학습을 안정화하기 위해 필요
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing

            # target_tensor가 없는 경우
            # inference 단계의 greedy decoding 수행
            else:

                # 현재 step에서 가장 점수가 높은 token 선택
                # 다음 step 입력으로 사용하기 위해 필요
                topv, topi = decoder_output.data.topk(1)

                # topk 결과 shape을 decoder 입력에 맞게 정리
                decoder_input = topi.squeeze(-1)

        # step별 출력을 sequence 차원으로 연결
        # [B, 1, OutVocab] 리스트 → [B, Seq, OutVocab]
        decoder_outputs = torch.cat(decoder_outputs, dim=1) # [B, Seq, OutVocab]

        # 출력 score를 log-probability로 변환
        # train.py에서 NLLLoss를 사용하므로 log_softmax 필요
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        # 전체 decoder 출력과 마지막 hidden state 반환
        # loss 계산과 후속 분석에 필요
        return decoder_outputs, decoder_hidden

    # decoder 한 step 연산
    # input: 현재 step 입력 token ID, shape [B, 1]
    # hidden: 이전 decoder hidden state, shape [1, B, D]
    def forward_step(self, input, hidden):

        # 입력 token ID를 embedding vector로 변환
        # GRU 입력은 정수 ID가 아닌 실수 벡터여야 하므로 필요
        output = self.embedding(input)

        # embedding에 ReLU 적용
        # 비선형성 부여 목적
        # 단, embedding 직후 ReLU는 필수 구조는 아니며 tutorial 스타일에 가까움
        output = F.relu(output)

        # decoder GRU 한 step 수행
        # 이전 hidden과 현재 입력 embedding을 이용해 다음 hidden 생성
        output, hidden = self.gru(output, hidden)

        # hidden/output을 출력 단어장 크기의 score로 변환
        # 다음 단어 후보 전체에 대한 점수 계산 목적
        output = self.out(output)

        # 현재 step 출력과 갱신된 hidden 반환
        return output, hidden


# Bahdanau Attention 클래스
# decoder가 매 출력 step마다 입력 문장의 어느 위치를 볼지 계산하는 모듈
# 고정된 encoder_hidden 하나에만 의존하는 기본 Seq2Seq의 정보 병목 완화 목적
class BahdanauAttention(nn.Module):

    # attention layer 초기화
    # hidden_size는 encoder hidden과 decoder hidden의 차원
    def __init__(self, hidden_size):

        # 부모 클래스 nn.Module 초기화
        # attention 내부 Linear layer 등록을 위해 필요
        super(BahdanauAttention, self).__init__()

        # decoder query를 attention score 계산 공간으로 투영하는 linear layer
        # query와 values를 같은 hidden_size 공간에서 더하기 위해 필요
        self.W1 = nn.Linear(hidden_size, hidden_size)

        # encoder values를 attention score 계산 공간으로 투영하는 linear layer
        # 각 입력 위치 hidden state를 query와 비교 가능하게 만들기 위해 필요
        self.W2 = nn.Linear(hidden_size, hidden_size)

        # tanh 결과를 scalar attention score로 바꾸는 linear layer
        # 각 입력 위치마다 하나의 중요도 점수 생성 목적
        self.V = nn.Linear(hidden_size, 1)

        # 추가 linear layer 선언
        # 현재 forward에서는 사용되지 않음
        # 불필요한 layer이며 제거 가능
        self.W3 = nn.Linear(hidden_size, 1)

    # attention forward 연산
    # query: decoder 현재 hidden state, shape [B, 1, D]
    # values: encoder_outputs, shape [B, M, D]
    # mask: 입력 문장 mask, shape [B, M]
    def forward(self, query, values, mask):

        # Additive attention
        # Bahdanau additive attention score 계산
        #
        # 수식:
        # score_ti = V(tanh(W1(query_t) + W2(value_i)))
        #
        # 필요한 이유:
        # - decoder 현재 상태 query와 encoder 각 위치 value의 관련성 계산
        # - 어떤 입력 단어를 많이 참고할지 결정
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values)))

        # scores shape 변환
        # 기존 shape: [B, M, 1]
        # 변경 shape: [B, 1, M]
        # softmax와 bmm 계산에서 attention weight 형태를 맞추기 위해 필요
        scores = scores.squeeze(2).unsqueeze(1) # [B, M, 1] -> [B, 1, M]

        # Dot-Product Attention: score(s_t, h_i) = s_t^T h_i
        # 아래 코드는 additive attention 대신 dot-product attention을 쓰고 싶을 때의 대안
        # query와 values의 내적으로 관련성 score 계산
        # Query [B, 1, D] * Values [B, D, M] -> Scores [B, 1, M]
        # scores = torch.bmm(query, values.permute(0,2,1))

        # Cosine Similarity: score(s_t, h_i) = cosine_similarity(s_t, h_i)
        # 아래 코드는 cosine similarity 기반 attention을 쓰고 싶을 때의 대안
        # 방향 유사도를 기준으로 관련성 score 계산
        # scores = F.cosine_similarity(query, values, dim=2).unsqueeze(1)

        # Mask out invalid positions.
        # padding 위치의 score를 -inf로 변경
        #
        # 필요한 이유:
        # - padding은 실제 단어가 아니므로 attention 대상이 되면 안 됨
        # - softmax(-inf)는 0에 가까운 weight가 되므로 padding 위치 무시 가능
        #
        # 주의:
        # - .data 사용은 autograd 관점에서 권장되지 않음
        # - scores = scores.masked_fill(mask.unsqueeze(1) == 0, -float('inf')) 권장
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))

        # Attention weights
        # score를 확률분포로 변환
        # 각 입력 위치에 대한 중요도 가중치 생성
        # shape: [B, 1, M]
        alphas = F.softmax(scores, dim=-1)

        # The context vector is the weighted sum of the values.
        # attention weight를 encoder_outputs에 곱해 context vector 계산
        #
        # 필요한 이유:
        # - decoder가 현재 step에서 참고할 입력 문장 요약 벡터 생성
        # - 모든 입력 위치를 동일하게 보는 것이 아니라 중요한 위치를 더 크게 반영
        context = torch.bmm(alphas, values)

        # context shape: [B, 1, D], alphas shape: [B, 1, M]
        # context: 현재 출력 단어 생성을 위한 입력 문맥 정보
        # alphas: attention weight 시각화나 분석에 사용 가능
        return context, alphas


# Attention Decoder 클래스
# Bahdanau attention을 사용해 출력 문장을 한 단어씩 생성하는 decoder
# 현재 EncoderDecoder에서 실제로 사용되는 decoder
class AttnDecoder(nn.Module):

    # Attention decoder 초기화
    # hidden_size: embedding/GRU hidden 차원
    # output_size: 출력 언어 vocabulary 크기
    def __init__(self, hidden_size, output_size):

        # 부모 클래스 nn.Module 초기화
        # 내부 layer와 parameter 등록을 위해 필요
        super(AttnDecoder, self).__init__()

        # 출력 token ID를 embedding vector로 변환하는 layer
        # decoder 입력은 이전 출력 단어 ID이므로 vector 변환 필요
        self.embedding = nn.Embedding(output_size, hidden_size)

        # Bahdanau attention 모듈
        # 매 step마다 encoder_outputs 중 중요한 입력 위치를 선택하기 위해 필요
        self.attention = BahdanauAttention(hidden_size)

        # decoder GRU
        # 입력 차원이 2 * hidden_size인 이유:
        # - 현재 decoder input embedding: hidden_size
        # - attention context vector: hidden_size
        # - 두 벡터 concat 결과: 2 * hidden_size
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)

        # GRU output을 출력 단어장 크기의 score로 변환
        # 다음 단어 후보 전체에 대한 점수 계산 목적
        self.out = nn.Linear(hidden_size, output_size)


    # Attention decoder 전체 forward
    # encoder_outputs: encoder의 모든 time step hidden state, shape [B, Seq, D]
    # encoder_hidden: encoder 마지막 hidden state, shape [1, B, D]
    # input_mask: 입력 문장 mask, shape [B, Seq]
    # target_tensor: 학습 시 teacher forcing에 사용
    # SOS_token: decoder 시작 token
    # max_len: 최대 출력 길이
    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):

        # Teacher forcing if given a target_tensor, otherwise greedy.
        # target_tensor가 있으면 학습 모드처럼 정답 이전 단어 사용
        # target_tensor가 없으면 추론 모드처럼 예측 단어 사용

        # batch size 추출
        # decoder_input 초기화를 위해 필요
        batch_size = encoder_outputs.size(0)

        # decoder 첫 입력 생성
        # 모든 sample의 첫 입력은 SOS_token
        # 문장 생성을 시작하는 신호로 필요
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)

        # decoder 초기 hidden state 설정
        # encoder 마지막 hidden state를 decoder 시작 상태로 사용
        # encoder와 decoder hidden dimension이 같기 때문에 직접 대입 가능
        # TODO는 encoder/decoder 구조가 달라질 경우 bridge layer 고려 의미
        decoder_hidden = encoder_hidden # TODO: Consider bridge

        # 각 step의 decoder output 저장 리스트
        # 모든 time step 출력을 나중에 하나의 sequence tensor로 결합하기 위해 필요
        decoder_outputs = []

        # max_len만큼 단어 생성 반복
        # 현재 코드는 EOS를 만나도 조기 중단하지 않고 고정 길이 생성
        for i in range(max_len):

            # attention을 포함한 decoder 한 step 수행
            # decoder_output: 현재 step의 output vocabulary score
            # decoder_hidden: 갱신된 decoder hidden state
            # attn_weights: 현재 step의 attention 분포
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, input_mask)

            # 현재 step 출력 저장
            # 최종적으로 [B, Seq, OutVocab] 형태를 만들기 위해 필요
            decoder_outputs.append(decoder_output)

            # target_tensor가 있는 경우
            # teacher forcing 사용
            if target_tensor is not None:

                # 정답 target의 i번째 token을 다음 decoder 입력으로 사용
                # 학습 초기에 모델 예측 오류가 다음 step으로 누적되는 문제 완화 목적
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing

            # target_tensor가 없는 경우
            # greedy decoding 사용
            else:

                # 현재 step에서 가장 score가 높은 token 선택
                # 추론 시 다음 step 입력으로 사용
                topv, topi = decoder_output.data.topk(1)

                # topk 결과를 다음 decoder 입력 shape으로 정리
                decoder_input = topi.squeeze(-1)

        # step별 output을 sequence 차원으로 결합
        # [B, 1, OutVocab] 여러 개 → [B, Seq, OutVocab]
        decoder_outputs = torch.cat(decoder_outputs, dim=1) # [B, Seq, OutVocab]

        # 출력 score를 log-probability로 변환
        # train.py의 NLLLoss 입력으로 사용하기 위해 필요
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        # 전체 decoder 출력과 마지막 hidden state 반환
        # decoder_outputs는 loss 계산에 사용
        # decoder_hidden은 필요한 경우 후속 분석에 사용 가능
        return decoder_outputs, decoder_hidden


    # attention decoder 한 step 연산
    # input: 현재 decoder 입력 token ID, shape [B, 1]
    # hidden: 현재 decoder hidden state, shape [1, B, D]
    # encoder_outputs: encoder 전체 출력, shape [B, Seq, D]
    # input_mask: 입력 mask, shape [B, Seq]
    def forward_step(self, input, hidden, encoder_outputs, input_mask):

        # encoder_outputs: [B, Seq, D]
        # 각 입력 위치의 encoder hidden state
        # attention values로 사용

        # hidden shape 변환
        # 기존 hidden shape: [1, B, D]
        # attention query shape 필요: [B, 1, D]
        # decoder 현재 상태를 encoder 각 위치와 비교하기 위해 필요
        query = hidden.permute(1, 0, 2) # [1, B, D] --> [B, 1, D]

        # attention context와 attention weights 계산
        # context: 현재 step에서 참고할 입력 문장 정보
        # attn_weights: 입력 위치별 중요도 분포
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)

        # 현재 decoder 입력 token ID를 embedding vector로 변환
        # GRU 입력으로 사용하기 위해 필요
        embedded = self.embedding(input)

        # 현재 입력 embedding과 attention context 결합
        #
        # 필요한 이유:
        # - embedded: 지금까지 생성된 출력 문장 쪽 정보
        # - context: 입력 문장에서 현재 참고해야 할 정보
        # - 두 정보를 함께 사용해야 다음 단어 예측 가능
        attn = torch.cat((embedded, context), dim=2)

        # 결합된 vector를 decoder GRU에 입력
        # 현재 step의 hidden state와 output 생성
        output, hidden = self.gru(attn, hidden)

        # GRU output을 출력 단어장 크기의 score로 변환
        # 각 출력 단어 후보에 대한 점수 계산
        output = self.out(output)

        # output: [B, 1, OutVocab]
        # 현재 step 출력 score, 갱신 hidden, attention weight 반환
        return output, hidden, attn_weights


# Encoder와 Decoder를 결합한 최종 Seq2Seq 모델 클래스
# train.py에서 실제로 생성하고 학습하는 모델
class EncoderDecoder(nn.Module):

    # 전체 모델 초기화
    # hidden_size: encoder/decoder 공통 hidden dimension
    # input_vocab_size: 입력 언어 vocabulary 크기
    # output_vocab_size: 출력 언어 vocabulary 크기
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size):

        # 부모 클래스 nn.Module 초기화
        # encoder와 decoder submodule 등록을 위해 필요
        super(EncoderDecoder, self).__init__()

        # Encoder 생성
        # 입력 문장을 encoder hidden representation으로 변환하기 위해 필요
        self.encoder = EncoderRNN(input_vocab_size, hidden_size)

        # Attention Decoder 생성
        # encoder_outputs를 attention으로 참고하며 출력 문장을 생성하기 위해 필요
        self.decoder = AttnDecoder(hidden_size, output_vocab_size)

        # Attention 없는 기본 decoder 사용 옵션
        # 현재는 주석 처리되어 있으며 사용되지 않음
        # attention 효과 비교 실험 시 아래 줄을 활성화 가능
        # self.decoder = DecoderRNN(hidden_size, output_vocab_size)

    # 전체 Encoder-Decoder forward
    # inputs: 입력 문장 token ID, shape [B, Seq]
    # input_mask: 입력 문장 mask, shape [B, Seq]
    # targets: 출력 정답 token ID, shape [B, Seq], 학습 시 teacher forcing에 사용
    def forward(self, inputs, input_mask, targets=None):

        # Encoder forward 수행
        # encoder_outputs: 모든 입력 위치의 hidden state
        # encoder_hidden: 마지막 hidden state
        #
        # 필요한 이유:
        # - encoder_outputs는 attention values로 사용
        # - encoder_hidden은 decoder 초기 hidden으로 사용
        encoder_outputs, encoder_hidden = self.encoder(inputs)

        # Decoder forward 수행
        # encoder 결과와 input_mask, target을 이용해 출력 문장 log-probability 생성
        #
        # targets가 있으면 teacher forcing 학습
        # targets가 없으면 greedy decoding 추론
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets)

        # decoder 출력과 마지막 hidden 반환
        # train.py에서 decoder_outputs를 loss 계산에 사용
        return decoder_outputs, decoder_hidden