import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EncoderRNN
# - 입력 시퀀스를 임베딩한 뒤 GRU로 인코딩
# - 매 타임스텝의 hidden state(output)와 최종 hidden state를 반환
# - AttnDecoder에서 output과 hidden 모두 활용
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        # output : [B, Seq, D] - 각 타임스텝의 hidden state (Attention의 values로 사용됨)
        # hidden : [1, B, D]   - 마지막 타임스텝의 hidden state (Decoder 초기 hidden으로 사용됨)
        return output, hidden


# DecoderRNN (Attention 없는 기본 디코더)
# - 인코더의 마지막 hidden state만을 초기 hidden으로 사용
# - 인코더 출력 전체를 참조하지 않으므로 긴 시퀀스에서 정보 손실이 발생 위험
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


# BahdanauAttention (Additive Attention)
# 핵심 수식: score(s_t, h_i) = V · tanh(W1·s_t + W2·h_i)

# [실행 순서]
# 1. Query(디코더 현재 hidden)와 Values(인코더 전체 출력)를 각각 선형 변환
#    - W1: query  [B, 1, D] → [B, 1, D]
#    - W2: values [B, M, D] → [B, M, D]
# 2. 두 결과를 더하고 tanh를 적용한 뒤, V로 스칼라 점수를 생성
#    - scores: [B, M, 1] → squeeze/unsqueeze → [B, 1, M]
# 3. 패딩 위치(mask==0)에 -inf를 채워 softmax 후 0이 되도록 함
# 4. softmax로 정규화하여 attention weight(alphas)를 얻음
# 5. alphas와 values를 행렬곱해 context vector를 생성
#    - context: [B, 1, D]  (디코더 스텝당 인코더 정보의 가중 합)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)  # query  변환 행렬
        self.W2 = nn.Linear(hidden_size, hidden_size)  # values 변환 행렬
        self.V = nn.Linear(hidden_size, 1)             # 스칼라 점수 산출
        self.W3 = nn.Linear(hidden_size, 1)            # (현재 미사용)

    def forward(self, query, values, mask):
        # Additive attention
        # query : [B, 1, D] - 디코더의 현재 hidden state
        # values: [B, M, D] - 인코더의 모든 타임스텝 hidden state
        # W1(query)  브로드캐스트: [B, 1, D] → [B, M, D] (M개 위치에 동일하게 더해짐)
        # W2(values): [B, M, D]
        # tanh(W1+W2): [B, M, D]
        # V(tanh(...)):  [B, M, 1]
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values)))
        scores = scores.squeeze(2).unsqueeze(1) # [B, M, 1] -> [B, 1, M]

        # Dot-Product Attention: score(s_t, h_i) = s_t^T h_i
        # Query [B, 1, D] * Values [B, D, M] -> Scores [B, 1, M]
        # scores = torch.bmm(query, values.permute(0,2,1))

        # Cosine Similarity: score(s_t, h_i) = cosine_similarity(s_t, h_i)
        # scores = F.cosine_similarity(query, values, dim=2).unsqueeze(1)

        # Mask out invalid positions.
        # 패딩 토큰(mask==0) 위치의 score를 -inf로 설정 → softmax 후 weight가 0이 됨
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))

        # Attention weights
        # softmax로 정규화: 모든 인코더 위치에 대한 중요도 분포 [B, 1, M]
        alphas = F.softmax(scores, dim=-1)

        # The context vector is the weighted sum of the values.
        # alphas [B, 1, M] × values [B, M, D] = context [B, 1, D]
        # → 현재 디코더 스텝에서 "어디에 집중할지"를 반영한 인코더 정보의 가중 합
        context = torch.bmm(alphas, values)

        # context shape: [B, 1, D], alphas shape: [B, 1, M]
        return context, alphas


# AttnDecoder (Attention이 적용된 디코더)
# - 매 디코딩 스텝마다 BahdanauAttention을 호출하여
#   인코더 출력 전체를 동적으로 참조한다.
# - GRU 입력 = [임베딩 || context vector] (2*hidden_size)
#   → 디코더가 현재 생성 중인 토큰과 관련 있는 인코더 위치에 집중하도록 유도한다.

class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        # GRU 입력 크기가 2*hidden_size인 이유:
        # 임베딩(hidden_size) + context vector(hidden_size)를 concat하기 때문
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):
        # Teacher forcing if given a target_tensor, otherwise greedy.
        batch_size = encoder_outputs.size(0)
        # 디코딩 시작: SOS(Start-Of-Sequence) 토큰으로 초기화
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden # TODO: Consider bridge
        decoder_outputs = []

        for i in range(max_len):
            # 매 스텝마다 attention을 새로 계산 > 인코더의 다른 위치를 동적으로 참조
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, input_mask)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Greedy decoding: 가장 확률 높은 토큰을 다음 입력으로 사용
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1) # [B, Seq, OutVocab]
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden


    def forward_step(self, input, hidden, encoder_outputs, input_mask):
        # encoder_outputs: [B, Seq, D]

        # 1. 현재 디코더 hidden state를 query로 변환: [1, B, D] → [B, 1, D]
        query = hidden.permute(1, 0, 2) # [1, B, D] --> [B, 1, D]

        # 2. Attention 계산
        #    query(현재 hidden)와 encoder_outputs(모든 인코더 hidden) 간의
        #    additive attention score를 계산하고 context vector를 생성한다.
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)

        # 3. 현재 토큰 임베딩: [B, 1, hidden_size]
        embedded = self.embedding(input)

        # 4. 임베딩과 context vector를 concat → GRU 입력: [B, 1, 2*hidden_size]
        #    context를 함께 넣음으로써 GRU가 인코더의 관련 정보를 직접 받아 처리한다.
        attn = torch.cat((embedded, context), dim=2)

        # 5. GRU 업데이트: 새로운 hidden state 생성
        output, hidden = self.gru(attn, hidden)

        # 6. 선형 변환으로 출력 어휘 크기로 매핑: [B, 1, output_vocab_size]
        output = self.out(output)
        # output: [B, 1, OutVocab]
        return output, hidden, attn_weights



# EncoderDecoder (전체 모델)
# - EncoderRNN + AttnDecoder를 조합한 Seq2Seq 모델
# - forward 흐름:
#   1. Encoder: 입력 시퀀스 (encoder_outputs, encoder_hidden)
#   2. AttnDecoder: 매 스텝마다 encoder_outputs 전체를 attention으로 참조하며 출력 생성
class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderRNN(input_vocab_size, hidden_size)
        self.decoder = AttnDecoder(hidden_size, output_vocab_size)
        # self.decoder = DecoderRNN(hidden_size, output_vocab_size)

    def forward(self, inputs, input_mask, targets=None):
        # inputs     : [B, Seq]      - 입력 토큰 인덱스
        # input_mask : [B, Seq]      - 패딩 위치 마스크 (1=유효, 0=패딩)
        # targets    : [B, Seq] or None - Teacher forcing용 정답 시퀀스
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets)
        return decoder_outputs, decoder_hidden