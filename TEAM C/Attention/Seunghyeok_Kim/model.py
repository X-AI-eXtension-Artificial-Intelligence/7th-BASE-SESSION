import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoder 정의
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):        # input_size: 입력 시퀀스의 단어 집합 크기, hidden_size: GRU의 hidden state 크기
        super(EncoderRNN, self).__init__()              # EncoderRNN 클래스의 초기화 함수로, nn.Module을 상속받아 초기화
        self.hidden_size = hidden_size                  # hidden_size를 클래스 변수로 저장

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input)            # Input 시퀀스에 대한 임베딩
        output, hidden = self.gru(embedded)         # GRU 모델을 사용한 Hidden, Output 연산
        return output, hidden


# Decoder 정의
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)             # 출력 시퀀스에 대한 임베딩
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)       # GRU 모델을 사용한 Hidden, Output 연산
        self.out = nn.Linear(hidden_size, output_size)                      # GRU의 출력에서 최종 출력으로 변환하는 선형 계층

    def forward(self, encoder_outputs, encoder_hidden, input_mask, target_tensor=None, SOS_token=0, max_len=10):    # Teacher forcing이 적용된 디코더의 forward 함수
        batch_size = encoder_outputs.size(0)                                                                        # 배치 크기 계산 : Encoder 출력의 첫 번째 차원 크기
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)                # 디코더 입력 초기화 : SOS 토큰으로 채워진 텐서 생성
        decoder_hidden = encoder_hidden                                                                             # 디코더의 초기 hidden state는 인코더의 마지막 hidden state로 설정
        decoder_outputs = []

        for i in range(max_len):                                                                        # 최대 시퀀스 길이만큼 반복
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)          # 디코더의 한 단계 forward 연산 수행
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:                                                               # Teacher forcing이 적용된 경우, 다음 디코더 입력은 타겟 시퀀스의 다음 토큰이 됨
                decoder_input = target_tensor[:, i].unsqueeze(1)                                        # 타겟 시퀀스에서 다음 토큰을 디코더 입력으로 설정
            else:
                topv, topi = decoder_output.data.topk(1)                                                # Teacher forcing이 적용되지 않은 경우, 다음 디코더 입력은 현재 디코더 출력에서 가장 높은 확률을 가진 토큰이 됨
                decoder_input = topi.squeeze(-1)                                                        # 디코더 출력에서 가장 높은 확률을 가진 토큰의 인덱스를 디코더 입력으로 설정                                            

        decoder_outputs = torch.cat(decoder_outputs, dim=1)                 # 디코더 출력들을 시퀀스 차원으로 연결하여 최종 디코더 출력 생성
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)            # 디코더 출력에 log softmax 적용하여 확률 분포로 변환
        return decoder_outputs, decoder_hidden                              # 최종 디코더 출력과 마지막 hidden state 반환

    def forward_step(self, input, hidden):                  # 디코더의 한 단계 forward 연산을 수행하는 함수
        output = self.embedding(input)                      # 입력 토큰에 대한 임베딩 계산
        output = F.relu(output)                             # ReLU 활성화 함수 적용
        output, hidden = self.gru(output, hidden)           # GRU 모델을 사용하여 다음 hidden state와 출력 계산
        output = self.out(output)                           # GRU의 출력을 최종 출력으로 변환하는 선형 계층 적용
        return output, hidden   


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)       # Query에 대한 선형 변환 계층
        self.W2 = nn.Linear(hidden_size, hidden_size)       # Values에 대한 선형 변환 계층
        self.V = nn.Linear(hidden_size, 1)                  # Attention score를 계산하는 선형 계층
        self.W3 = nn.Linear(hidden_size, 1)                 # Masking을 위한 선형 계층 (선택적)

    def forward(self, query, values, mask):
        # Additive attention
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values)))       # Query와 Values에 대한 선형 변환을 적용한 후, tanh 활성화 함수를 거쳐 Attention score 계산
        scores = scores.squeeze(2).unsqueeze(1)                             # Attention score의 차원을 조정하여 [B, 1, M] 형태로 만듦

        # Dot-Product Attention: score(s_t, h_i) = s_t^T h_i
        # Query [B, 1, D] * Values [B, D, M] -> Scores [B, 1, M]
        # scores = torch.bmm(query, values.permute(0,2,1))

        # Cosine Similarity: score(s_t, h_i) = cosine_similarity(s_t, h_i)
        # scores = F.cosine_similarity(query, values, dim=2).unsqueeze(1)

        # Mask out invalid positions.
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))     # Masking을 적용하여 유효하지 않은 위치의 Attention score를 -inf로 설정하여 softmax에서 0이 되도록 함

        # Attention weights
        alphas = F.softmax(scores, dim=-1)      # Attention score에 softmax를 적용하여 Attention weights 계산 (각 값이 전체 값에 대한 중요도를 나타냄)

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, values)     # Attention weights와 Values의 가중합을 계산하여 Context vector 생성 (각 값이 전체 값에 대한 중요도를 반영하여 합산됨)

        # context shape: [B, 1, D], alphas shape: [B, 1, M]
        return context, alphas                  # Context vector와 Attention weights 반환


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttnDecoder, self).__init__()                                     # AttnDecoder 클래스의 초기화 함수로, nn.Module을 상속받아 초기화
        self.embedding = nn.Embedding(output_size, hidden_size)                 # 출력 시퀀스에 대한 임베딩 계층 정의
        self.attention = BahdanauAttention(hidden_size)                         # Bahdanau Attention 메커니즘을 사용하는 Attention 계층 정의
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)       # GRU 모델을 사용하여 Hidden, Output 연산 정의 (입력 크기는 임베딩된 입력과 Attention context의 크기를 합친 것)
        self.out = nn.Linear(hidden_size, output_size)                          # GRU의 출력에서 최종 출력으로 변환하는 선형 계층 정의


    def forward(self, encoder_outputs, encoder_hidden, input_mask, target_tensor=None, SOS_token=0, max_len=10):    # Teacher forcing이 적용된 디코더의 forward 함수
        batch_size = encoder_outputs.size(0)                                                                        # 배치 크기 계산 : Encoder 출력의 첫 번째 차원 크기
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)                # 디코더 입력 초기화 : SOS 토큰으로 채워진 텐서 생성
        decoder_hidden = encoder_hidden                                                                             # 디코더의 초기 hidden state는 인코더의 마지막 hidden state로 설정
        decoder_outputs = []

        for i in range(max_len):                                                                    # 최대 시퀀스 길이만큼 반복
            decoder_output, decoder_hidden, attn_weights = self.forward_step(                       
                decoder_input, decoder_hidden, encoder_outputs, input_mask)                         # 디코더의 한 단계 forward 연산 수행하여 디코더 출력, 다음 hidden state, Attention weights 계산
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:                                                  # Teacher forcing이 적용된 경우, 다음 디코더 입력은 타겟 시퀀스의 다음 토큰이 됨
                decoder_input = target_tensor[:, i].unsqueeze(1)                           # Teacher forcing
            else:
                topv, topi = decoder_output.data.topk(1)                                   # Teacher forcing이 적용되지 않은 경우, 다음 디코더 입력은 현재 디코더 출력에서 가장 높은 확률을 가진 토큰이 됨
                decoder_input = topi.squeeze(-1)                                           # 디코더 출력에서 가장 높은 확률을 가진 토큰의 인덱스를 디코더 입력으로 설정

        decoder_outputs = torch.cat(decoder_outputs, dim=1)                 # 디코더 출력들을 시퀀스 차원으로 연결하여 최종 디코더 출력 생성
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)            # 디코더 출력에 log softmax 적용하여 확률 분포로 변환
        return decoder_outputs, decoder_hidden


    def forward_step(self, input, hidden, encoder_outputs, input_mask):                 # 디코더의 한 단계 forward 연산을 수행하는 함수
        query = hidden.permute(1, 0, 2)                                                 # [1, B, D] --> [B, 1, D]
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)      # Attention 메커니즘을 사용하여 Context vector와 Attention weights 계산
        embedded = self.embedding(input)                                                # 입력 토큰에 대한 임베딩 계산
        attn = torch.cat((embedded, context), dim=2)                                    # 임베딩된 입력과 Attention context를 연결하여 GRU의 입력으로 사용
        output, hidden = self.gru(attn, hidden)                                         # GRU 모델을 사용하여 다음 hidden state와 출력 계산
        output = self.out(output)                                                       # GRU의 출력을 최종 출력으로 변환하는 선형 계층 적용
        return output, hidden, attn_weights


class EncoderDecoder(nn.Module):                                                # Encoder와 Decoder를 통합하는 모델 클래스 정의
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size):       # EncoderDecoder 클래스의 초기화 함수로, nn.Module을 상속받아 초기화
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderRNN(input_vocab_size, hidden_size)                # EncoderRNN 인스턴스를 생성하여 Encoder 계층 정의
        self.decoder = AttnDecoder(hidden_size, output_vocab_size)              # AttnDecoder 인스턴스를 생성하여 Decoder 계층 정의 

    def forward(self, inputs, input_mask, targets=None):                        # 모델의 forward 함수로, 입력 시퀀스, 입력 마스크, 타겟 시퀀스를 받아서 디코더 출력과 마지막 hidden state를 반환
        encoder_outputs, encoder_hidden = self.encoder(inputs)                  # Encoder를 사용하여 입력 시퀀스에 대한 출력과 마지막 hidden state 계산
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets)               # Decoder를 사용하여 디코더 출력과 마지막 hidden state 계산 (Teacher forcing이 적용된 경우 타겟 시퀀스도 전달)
        return decoder_outputs, decoder_hidden
