import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module): # 입력 문장을 읽는 encoder
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size) # 단어 번호를 벡터로 만듦, 고정 벡터가 아니기에 hiddenszie의 크기의 벡터로
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True) # 이 gru모델 문장이 앞에서부터 순서대로 읽으면서 문맥정보 저장

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded) # 여기서 output이 입력문장의 각 단어 위치마다 나온 hidden_size
        return output, hidden
        # 인코더는 입력 문장을 embedding한 뒤 GRU에 넣음
        # GRU는 각 단어 위치에 대한 출력값인 encoder_outputs와 문장 전체 정보를 담은 마지막 hidden state인 encoder_hidden을 반환


class DecoderRNN(nn.Module): # attention이 없는 기본 디코더
    # Standard non-attentional decoder
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, input_mask, # encoder_hidden이 인코더의 마지막 hidden을 뜻함
                target_tensor=None, SOS_token=0, max_len=10):
        # Teacher forcing if given a target_tensor, otherwise greedy.
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden # TODO: Consider bridge
        decoder_outputs = []

        for i in range(max_len):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None: # 정답 단어를 다음 입력으로 넣어줌 
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1) # [B, Seq, OutVocab]
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden
        # 정답 문장이 주어지면 정답 토큰을 다음 입력으로 넣고, 추론할 때는 모델이 예측한 단어를 다시 입력으로 사용
    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

        # 여기가 attention을 직접 만드는 코드임
class BahdanauAttention(nn.Module): #디코더가 출력 단어를 만들 때 입력 문장의 어느 단어를 집중해서 볼지 계산하는 방식
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.W3 = nn.Linear(hidden_size, 1)

    def forward(self, query, values, mask): # 현대 디코더 hidden_state, 입력 문장의 각 단어 정보
        # Additive attention
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values))) #attention 공식 여기가 각 단어가 얼마나 중요한지 점수 계산
        scores = scores.squeeze(2).unsqueeze(1) # [B, M, 1] -> [B, 1, M]

        # Dot-Product Attention: score(s_t, h_i) = s_t^T h_i
        # Query [B, 1, D] * Values [B, D, M] -> Scores [B, 1, M]
        # scores = torch.bmm(query, values.permute(0,2,1))

        # Cosine Similarity: score(s_t, h_i) = cosine_similarity(s_t, h_i)
        # scores = F.cosine_similarity(query, values, dim=2).unsqueeze(1)

        # Mask out invalid positions.
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))

        # Attention weights
        alphas = F.softmax(scores, dim=-1) # 확률로 볼 수 있게끔 바꿈

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, values) # 인코더 출력값들의 가중합을 해주는 코드

        # context shape: [B, 1, D], alphas shape: [B, 1, M]
        return context, alphas


class AttnDecoder(nn.Module): # attention이 적용된 코드
    def __init__(self, hidden_size, output_size):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True) # embeding+context vector이렇게 두개를 써야하기에 2*hiddensize를 쓴거임
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):
        # Teacher forcing if given a target_tensor, otherwise greedy.
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden # TODO: Consider bridge
        decoder_outputs = []

        for i in range(max_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, input_mask)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1) # [B, Seq, OutVocab]
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden


    def forward_step(self, input, hidden, encoder_outputs, input_mask):
        # encoder_outputs: [B, Seq, D]
        query = hidden.permute(1, 0, 2) # [1, B, D] --> [B, 1, D] attention계산을 위해 바꿔줌
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)
        embedded = self.embedding(input)
        attn = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(attn, hidden)
        output = self.out(output) # 여기서가 다음 단어가 무엇인지에 대한 점수를 만드는 코드
        # output: [B, 1, OutVocab]
        return output, hidden, attn_weights


class EncoderDecoder(nn.Module): # 인코더와 디코더를 하나로 묶은 최종 모델
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderRNN(input_vocab_size, hidden_size)
        self.decoder = AttnDecoder(hidden_size, output_vocab_size)
        # self.decoder = DecoderRNN(hidden_size, output_vocab_size)

    def forward(self, inputs, input_mask, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets)
        return decoder_outputs, decoder_hidden
