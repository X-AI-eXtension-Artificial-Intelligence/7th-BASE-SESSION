import torch
import torch.nn as nn
# 파이토치에서 딥러닝 모델을 구축하고 학습시키기 위한 핵심 모듈
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    # nn.Mudule: 딥러닝 모델의 거대한 클래스
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # nn.GRU: 시계열/언어 데이터를 위한 순환 신경망층

    def forward(self, input):
        # 입력 문장을 임베딩 벡터로 변환
        embedded = self.embedding(input)
        # GRU를 거쳐 전체 문장의 문맥 정보를 담은 출력(output)과 마지막 상태(hidden) 반환
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    # Standard non-attentional decoder
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

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


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size) # Query(디코더 hidden) 변환
        # nn.Linear: 완전 연결 층(입력값과 가중치를 곱하고 편향을 더하는 기초 연산)
        self.W2 = nn.Linear(hidden_size, hidden_size) # Values(인코더 outputs) 변환
        self.V = nn.Linear(hidden_size, 1) # 점수 산출을 위한 최종 가중치
        self.W3 = nn.Linear(hidden_size, 1)

    def forward(self, query, values, mask):
        # Additive attention
        # 스코어 계산: 쿼리(현재 상태)와 밸류(인코더 출력)의 연관성 측정
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values)))
        scores = scores.squeeze(2).unsqueeze(1) # [B, M, 1] -> [B, 1, M]
        # squeeze로 마지막 차원의 크기가 1인 것을 제거한 후
        # unsqeeze로 인덱스 1번 위치에 새로운 차원(크기 1)을 추가함

        # Dot-Product Attention: score(s_t, h_i) = s_t^T h_i
        # Query [B, 1, D] * Values [B, D, M] -> Scores [B, 1, M]
        # scores = torch.bmm(query, values.permute(0,2,1))

        # Cosine Similarity: score(s_t, h_i) = cosine_similarity(s_t, h_i)
        # scores = F.cosine_similarity(query, values, dim=2).unsqueeze(1)

        # Mask out invalid positions.
        # 마스킹: 패딩 부분(문장이 아닌 곳)은 무시하도록 -inf 처리
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))

        # Attention weights
        # 소프트맥스: 어텐션 가중치(alphas) 생성
        alphas = F.softmax(scores, dim=-1)

        # The context vector is the weighted sum of the values.
        # 컨텍스트 벡터 생성: 가중합(Weighted Sum)을 통해 집중해야 할 정보 압축
        context = torch.bmm(alphas, values)

        # context shape: [B, 1, D], alphas shape: [B, 1, M]
        return context, alphas


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        # 어텐션 컨텍스트와 임베딩을 합치므로 2 * hidden_size가 입력으로 들어감
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
        # 현재 상태(hidden)를 Query로 하여 어텐션 수행
        query = hidden.permute(1, 0, 2) # [1, B, D] --> [B, 1, D]
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)
        embedded = self.embedding(input)
        # 임베딩 정보와 어텐션이 추출한 컨텍스트 정보를 병합(concatenate)
        attn = torch.cat((embedded, context), dim=2)
        # 병합된 정보를 GRU에 넣어 다음 단어 예측
        output, hidden = self.gru(attn, hidden)
        output = self.out(output)
        # output: [B, 1, OutVocab]
        return output, hidden, attn_weights


class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size):
        super(EncoderDecoder, self).__init__()
        # 인코더: 입력 문장(소스)을 정보가 압축된 벡터로 변환
        self.encoder = EncoderRNN(input_vocab_size, hidden_size)
        # 디코더: 압축된 정보를 기반으로 타겟 문장 생성
        self.decoder = AttnDecoder(hidden_size, output_vocab_size)
        # self.decoder = DecoderRNN(hidden_size, output_vocab_size)

    def forward(self, inputs, input_mask, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets)
        return decoder_outputs, decoder_hidden
