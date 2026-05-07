import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 인코더 (Encoder) ---
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # 단어 인덱스를 임베딩 벡터로 변환
        self.embedding = nn.Embedding(input_size, hidden_size)
        # GRU 셀을 사용하여 시퀀스를 순차적으로 처리
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input)
        # 출력값(output)과 문맥 정보가 압축된 마지막 은닉 상태(hidden) 반환
        output, hidden = self.gru(embedded)
        return output, hidden

# --- 2. 기본 디코더 (어텐션 미사용) ---
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size) # 단어 집합 크기로 변환

    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):
        batch_size = encoder_outputs.size(0)
        # 첫 입력으로 SOS 토큰 제공
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden # 인코더의 마지막 상태를 디코더의 초기 상태로 사용
        decoder_outputs = []

        # 최대 길이(max_len)만큼 단어를 하나씩 생성
        for i in range(max_len):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            # 교사 강요 (Teacher Forcing): 정답 타겟이 주어지면 이전 예측값 대신 실제 정답을 다음 입력으로 사용
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # 타겟이 없으면(추론 시), 모델이 가장 높게 예측한 단어를 다음 입력으로 사용 (Greedy)
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        # 결과물 병합 및 확률값 계산 (Log Softmax)
        decoder_outputs = torch.cat(decoder_outputs, dim=1) 
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden):
        # 1-step 계산 (단일 타임스텝)
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

# --- 3. 어텐션 레이어 (Bahdanau 어텐션) ---
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        # 가중치 계산을 위한 선형 변환 레이어들
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.W3 = nn.Linear(hidden_size, 1)

    def forward(self, query, values, mask):
        # Additive attention 방식으로 어텐션 스코어 계산
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values)))
        scores = scores.squeeze(2).unsqueeze(1) 

        # 패딩된 부분(mask 값이 0)은 어텐션을 주지 않도록 -inf로 마스킹 처리
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))

        # Softmax를 통과시켜 확률값(Attention Weights) 도출
        alphas = F.softmax(scores, dim=-1)

        # 인코더의 출력값(values)에 가중치(alphas)를 곱하여 문맥 벡터(Context Vector) 생성
        context = torch.bmm(alphas, values)
        return context, alphas

# --- 4. 어텐션 디코더 ---
class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        # 이전 출력, 문맥 벡터를 함께 받으므로 입력 크기가 2 * hidden_size
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    # forward() 로직은 DecoderRNN과 거의 동일하게 순차적으로 단어를 생성
    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):
        # ... (DecoderRNN의 forward와 동일한 루프 구조) ...
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden 
        decoder_outputs = []

        for i in range(max_len):
            # forward_step 호출 시 어텐션을 위해 encoder_outputs와 mask를 함께 넘김
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, input_mask)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1) 
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden, encoder_outputs, input_mask):
        # 디코더의 현재 은닉 상태를 쿼리(Query)로 사용
        query = hidden.permute(1, 0, 2)
        # 어텐션 메커니즘을 통해 문맥 벡터 생성
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)
        
        embedded = self.embedding(input)
        # 임베딩된 입력과 문맥 벡터를 연결(Concatenate)하여 GRU에 입력
        attn = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(attn, hidden)
        output = self.out(output)
        return output, hidden, attn_weights

# --- 5. 전체 Seq2Seq 래퍼 모델 (Encoder + Decoder) ---
class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderRNN(input_vocab_size, hidden_size)
        self.decoder = AttnDecoder(hidden_size, output_vocab_size) # 기본적으로 어텐션 디코더 사용

    def forward(self, inputs, input_mask, targets=None):
        # 1. 인코더 통과
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        # 2. 디코더 통과 (어텐션을 위해 인코더 출력과 마스크 정보 전달)
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets)
        return decoder_outputs, decoder_hidden