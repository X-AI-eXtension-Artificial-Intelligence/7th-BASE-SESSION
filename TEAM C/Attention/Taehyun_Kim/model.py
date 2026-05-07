"""
model.py
========
Seq2Seq + Bahdanau Attention 모델 구현.

전체 구조:
  [입력 시퀀스]
       │
  EncoderRNN          ← 입력 시퀀스를 hidden state 시퀀스로 인코딩
       │  encoder_outputs (모든 스텝), encoder_hidden (마지막 스텝)
       ▼
  BahdanauAttention   ← 디코더 쿼리와 인코더 출력 사이의 어텐션 가중치 계산
       │  context vector (가중 합산)
       ▼
  AttnDecoder         ← context + 이전 출력을 입력으로 한 스텝씩 디코딩
       │
  [출력 시퀀스]
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────
# 1. EncoderRNN
# ──────────────────────────────────────────
class EncoderRNN(nn.Module):
    """
    입력 시퀀스를 GRU로 인코딩하여 hidden state 시퀀스를 생성.
    batch size: 한 번에 처리하는 문장 수
    SeqLensequence length: 문장의 최대 토큰 수 (MAX_LENGTH)
    Hhidden size: 각 단어/hidden을 표현하는 벡터 차원

    입력:
        input: [B, SeqLen]  (패딩된 인덱스 시퀀스)
    출력:
        output: [B, SeqLen, H]  (각 타임스텝의 hidden state)
        hidden: [1, B, H]       (마지막 타임스텝의 hidden state)
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # 단어 인덱스 -> 밀집 벡터 변환 (input_size: 입력 어휘 크기)
        self.embedding = nn.Embedding(input_size, hidden_size)

        # GRU: hidden_size 입력 -> hidden_size 출력
        # batch_first=True: 입력/출력 shape를 [B, Seq, H]로 통일
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # GRU(Gated Recurrent Unit): 시퀀스 데이터를 순서대로 처리하면서 "기억"을 유지하는 신경망

    def forward(self, input):
        # [B, SeqLen] -> [B, SeqLen, H]
        embedded = self.embedding(input)

        # output: [B, SeqLen, H] - 모든 타임스텝의 hidden state
        # hidden: [1, B, H]      - 마지막 타임스텝의 hidden state
        output, hidden = self.gru(embedded)
        return output, hidden


# ──────────────────────────────────────────
# 2. DecoderRNN (어텐션 없는 기본 디코더, 참고용)
# ──────────────────────────────────────────
class DecoderRNN(nn.Module):
    """
    어텐션 없는 기본 디코더.
    인코더의 마지막 hidden state만 이용해 디코딩.
    (AttnDecoder와 비교용으로 남겨둠)

    Teacher Forcing:
        학습 시 target_tensor가 주어지면 실제 정답 토큰을 다음 입력으로 사용.
        추론 시에는 이전 예측 토큰을 다음 입력으로 사용 (greedy).
    """
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru        = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out        = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):
        batch_size = encoder_outputs.size(0)

        # 디코더 첫 입력: SOS 토큰  [B, 1]
        decoder_input  = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden   # 인코더 마지막 hidden을 초기 hidden으로 사용
        decoder_outputs = []

        for i in range(max_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher Forcing: 정답 토큰을 다음 입력으로
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # Greedy: 가장 높은 확률 토큰을 다음 입력으로
                _, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        # 각 스텝 출력을 시퀀스 차원으로 연결: [B, SeqLen, OutVocab]
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # Log-softmax로 변환 (NLLLoss와 함께 사용)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden):
        """단일 타임스텝 디코딩"""
        output = self.embedding(input)           # [B, 1, H]
        output = F.relu(output)                  # 비선형 활성화
        output, hidden = self.gru(output, hidden)
        output = self.out(output)                # [B, 1, OutVocab]
        return output, hidden


# ──────────────────────────────────────────
# 3. BahdanauAttention (핵심 어텐션 모듈)
# ──────────────────────────────────────────
class BahdanauAttention(nn.Module):
    """
    Additive Attention (Bahdanau et al., 2015).

    디코더의 현재 hidden state(query)와 인코더 전체 출력(values) 사이의
    어텐션 점수를 계산하고, 가중 합산된 context 벡터를 반환.

    수식:
        score(s_t, h_i) = V · tanh(W1·s_t + W2·h_i)
        alpha_i = softmax(score_i)
        context = Σ alpha_i · h_i

    비교 가능한 다른 어텐션 방식 (주석으로 포함):
        Dot-Product:  score = s_t^T · h_i
        Cosine:       score = cosine_similarity(s_t, h_i)
    """
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        # W1: 쿼리(디코더 hidden) 변환 선형층
        self.W1 = nn.Linear(hidden_size, hidden_size)
        # W2: 값(인코더 출력) 변환 선형층
        self.W2 = nn.Linear(hidden_size, hidden_size)
        # V: 어텐션 점수를 스칼라로 압축하는 선형층
        self.V  = nn.Linear(hidden_size, 1)
        # W3: (미사용) 예비 선형층
        self.W3 = nn.Linear(hidden_size, 1)

    def forward(self, query, values, mask):
        """
        Args:
            query:  [B, 1, H]        - 디코더 현재 hidden state
            values: [B, SeqLen, H]   - 인코더 전체 hidden states
            mask:   [B, SeqLen]      - 패딩 마스크 (유효=1, 패딩=0)
        Returns:
            context: [B, 1, H]       - 어텐션 가중 합산 벡터
            alphas:  [B, 1, SeqLen]  - 어텐션 가중치
        """
        # Additive attention 점수 계산
        # W1(query): [B, 1, H]  브로드캐스팅으로 W2(values): [B, SeqLen, H]와 합산
        # scores: [B, SeqLen, 1]
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values)))

        # [B, SeqLen, 1] -> [B, 1, SeqLen] (softmax를 마지막 dim에 적용하기 위해)
        scores = scores.squeeze(2).unsqueeze(1)

        # 패딩 위치의 점수를 -inf로 설정 → softmax 후 0이 됨
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))

        # Softmax로 어텐션 가중치 계산: [B, 1, SeqLen]
        alphas = F.softmax(scores, dim=-1)

        # 가중 합산: context = Σ alpha_i * h_i
        # alphas: [B, 1, SeqLen] × values: [B, SeqLen, H] -> context: [B, 1, H]
        context = torch.bmm(alphas, values)

        return context, alphas


# ──────────────────────────────────────────
# 4. AttnDecoder (어텐션 적용 디코더)
# ──────────────────────────────────────────
class AttnDecoder(nn.Module):
    """
    Bahdanau Attention을 사용하는 디코더.

    각 디코딩 스텝에서:
    1. 현재 hidden으로 인코더 출력 전체에 어텐션 계산
    2. context 벡터와 임베딩을 결합해 GRU 입력 생성
    3. GRU 출력으로 다음 토큰 예측

    GRU 입력 크기가 2*hidden_size인 이유:
        [embedding(hidden_size) || context(hidden_size)] = 2*hidden_size
    """
    def __init__(self, hidden_size, output_size):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        # context와 임베딩을 concat하므로 입력 크기 2*hidden_size
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):
        """
        Args:
            encoder_outputs: [B, SeqLen, H]  - 인코더 전체 출력
            encoder_hidden:  [1, B, H]       - 인코더 마지막 hidden
            input_mask:      [B, SeqLen]     - 입력 패딩 마스크
            target_tensor:   [B, SeqLen]     - Teacher Forcing용 정답 (추론 시 None)
        Returns:
            decoder_outputs: [B, SeqLen, OutVocab]  - log-softmax 확률
            decoder_hidden:  [1, B, H]
        """
        batch_size = encoder_outputs.size(0)

        # 시작 토큰 [B, 1]
        decoder_input  = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(max_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, input_mask
            )
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher Forcing
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # Greedy Decoding
                _, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)   # [B, SeqLen, OutVocab]
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden, encoder_outputs, input_mask):
        """
        단일 타임스텝 어텐션 디코딩.

        Args:
            input:           [B, 1]          - 현재 입력 토큰
            hidden:          [1, B, H]       - 현재 디코더 hidden
            encoder_outputs: [B, SeqLen, H]  - 인코더 전체 출력
            input_mask:      [B, SeqLen]     - 패딩 마스크
        Returns:
            output:      [B, 1, OutVocab]
            hidden:      [1, B, H]
            attn_weights:[B, 1, SeqLen]
        """
        # hidden: [1, B, H] -> [B, 1, H] (어텐션 query 형태로 변환)
        query = hidden.permute(1, 0, 2)

        # 어텐션 계산: context [B, 1, H], attn_weights [B, 1, SeqLen]
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)

        # 현재 토큰 임베딩: [B, 1, H]
        embedded = self.embedding(input)

        # 임베딩 + context 결합: [B, 1, 2H]
        attn = torch.cat((embedded, context), dim=2)

        # GRU 업데이트
        output, hidden = self.gru(attn, hidden)

        # 출력 어휘 크기로 projection: [B, 1, OutVocab]
        output = self.out(output)

        return output, hidden, attn_weights


# ──────────────────────────────────────────
# 5. EncoderDecoder (전체 Seq2Seq 모델)
# ──────────────────────────────────────────
class EncoderDecoder(nn.Module):
    """
    EncoderRNN + AttnDecoder를 묶은 최상위 Seq2Seq 모델.

    forward 흐름:
        inputs -> Encoder -> (encoder_outputs, encoder_hidden)
                          -> AttnDecoder -> decoder_outputs
    """
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderRNN(input_vocab_size, hidden_size)
        self.decoder = AttnDecoder(hidden_size, output_vocab_size)
        # 어텐션 없는 디코더로 교체하려면 아래 주석 해제:
        # self.decoder = DecoderRNN(hidden_size, output_vocab_size)

    def forward(self, inputs, input_mask, targets=None):
        """
        Args:
            inputs:     [B, SeqLen]  - 패딩된 입력 인덱스
            input_mask: [B, SeqLen]  - 패딩 마스크
            targets:    [B, SeqLen]  - Teacher Forcing 정답 (추론 시 None)
        Returns:
            decoder_outputs: [B, SeqLen, OutVocab]
            decoder_hidden:  [1, B, H]
        """
        # 인코딩
        encoder_outputs, encoder_hidden = self.encoder(inputs)

        # 디코딩 (어텐션 포함)
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets
        )
        return decoder_outputs, decoder_hidden
