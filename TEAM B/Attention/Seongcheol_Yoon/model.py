import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):  # EncoderRNN은 입력 문장을 읽어서 hidden state로 바꾸는 역할을 합니다.
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size  # 256

        self.embedding = nn.Embedding(input_size, hidden_size) # Embedding은 단어 ID를 벡터로 바꿔주는 층
        
        # GRU는 RNN의 한 종류로 
        # 문장을 단어 순서대로 읽으면서 hidden state를 업데이트
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        return output, hidden 
    # GRU는 각 단어 위치에 대한 hidden state인 encoder_outputs와 마지막 hidden state인 encoder_hidden을 반환합니다. 
    # encoder_outputs는 Attention이 입력 문장의 어느 단어를 참고할지 계산할 때 사용되고, encoder_hidden은 decoder의 초기 hidden state로 사용됩니다.

class DecoderRNN(nn.Module): # Attention을 사용하지 않는 기본 Decoder
    # Standard non-attentional decoder 
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__() 
        self.embedding = nn.Embedding(output_size, hidden_size) # 출력 언어의 단어 ID를 hidden_size 차원의 벡터로 변환합
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True) # GRU는 RNN의 한 종류로 문장을 단어 순서대로 읽으면서 hidden state를 업데이트
        self.out = nn.Linear(hidden_size, output_size) # 출력 언어의 단어 ID를 output_size 차원의 벡터로 변환합

# 출력 문장을 생성하는 함수
    def forward(self, encoder_outputs, encoder_hidden, input_mask, 
                target_tensor=None, SOS_token=0, max_len=10):
        # Teacher forcing if given a target_tensor, otherwise greedy.
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden # TODO: Consider bridge
        decoder_outputs = []

        for i in range(max_len):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1) # [B, Seq, OutVocab]
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

# BahdanauAttention은 Attention 층으로 입력 문장의 각 단어를 참고할 때 가중치를 계산하는 역할을 합니다.
class BahdanauAttention(nn.Module):
    # Decoder가 다음 단어를 예측할 때 입력 문장의 어느 위치를 더 참고할지 계산합니다.
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size) # query, 즉 현재 decoder hidden state를 변환하는 선형층입니다.
        self.W2 = nn.Linear(hidden_size, hidden_size) # values, 즉 입력 문장의 각 단어를 변환하는 선형층입니다.
        self.V = nn.Linear(hidden_size, 1) # 최종 점수를 계산하는 선형층입니다.
        self.W3 = nn.Linear(hidden_size, 1)  # ?

    def forward(self, query, values, mask):
        # Additive attention
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values)))
        scores = scores.squeeze(2).unsqueeze(1) # [B, M, 1] -> [B, 1, M]

        # Mask out invalid positions.
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))

        # Attention weights
        alphas = F.softmax(scores, dim=-1)

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, values)

        # context shape: [B, 1, D], alphas shape: [B, 1, M]
        return context, alphas


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
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
        query = hidden.permute(1, 0, 2) # [1, B, D] --> [B, 1, D]
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)
        embedded = self.embedding(input)
        attn = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(attn, hidden)
        output = self.out(output)
        # output: [B, 1, OutVocab]
        return output, hidden, attn_weights


class EncoderDecoder(nn.Module):
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
