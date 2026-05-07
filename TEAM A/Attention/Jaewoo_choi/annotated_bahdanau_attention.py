"""
annotated_bahdanau_attention.py

mhauskn/pytorch_attention 예제 구조를 기반으로 정리한 Bahdanau Attention 실습 코드다.

이 파일은 과제 제출용으로, 모델의 핵심 구조와 이론적 의미를 주석으로 설명한다.

핵심 참고:
- Bahdanau, Cho, Bengio, 2015, Neural Machine Translation by Jointly Learning to Align and Translate
- mhauskn/pytorch_attention
- PyTorch seq2seq translation tutorial 계열 데이터 구성
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
MAX_LENGTH = 10


class EncoderRNN(nn.Module):
    """
    GRU 기반 Encoder다.

    기존 encoder-decoder 구조에서는 source sentence 전체를 마지막 hidden state 하나에 압축한다.
    하지만 Bahdanau attention에서는 encoder의 각 시점 hidden state 전체를 decoder가 다시 참조한다.

    따라서 이 encoder의 outputs는 단순한 중간 결과가 아니라,
    decoder가 alignment를 학습하기 위한 source-side memory 역할을 한다.
    """

    def __init__(self, input_vocab_size, hidden_size, pad_idx=PAD_TOKEN):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, hidden_size, padding_idx=pad_idx)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class BahdanauAttention(nn.Module):
    """
    Bahdanau Additive Attention 모듈이다.

    원 논문의 alignment model은 다음 형태로 이해할 수 있다.

    e_{t,i} = v_a^T tanh(W_a s_{t-1} + U_a h_i)

    여기서 e_{t,i}는 target 시점 t에서 source 위치 i가 얼마나 중요한지를 나타내는 score다.
    s_{t-1}는 decoder hidden state이고, h_i는 encoder의 i번째 hidden state다.

    softmax를 적용하면 attention weight alpha_{t,i}가 된다.

    alpha_{t,i} = softmax(e_{t,i})

    이후 context vector는 다음처럼 계산된다.

    c_t = sum_i alpha_{t,i} h_i

    즉, attention은 번역 과정에서 source 문장의 어느 부분을 볼지 동적으로 결정하는 장치다.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_values = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, query, values, mask):
        scores = self.V(torch.tanh(self.W_query(query) + self.W_values(values)))
        scores = scores.squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        alphas = F.softmax(scores, dim=-1)
        context = torch.bmm(alphas.unsqueeze(1), values)
        return context, alphas


class AttnDecoderRNN(nn.Module):
    """
    Bahdanau attention을 포함한 Decoder다.

    매 decoding step에서 decoder는 다음 작업을 수행한다.
    1. 이전 token을 embedding한다.
    2. decoder hidden state를 query로 사용한다.
    3. encoder outputs 전체에 대해 attention score를 계산한다.
    4. attention weight로 context vector를 만든다.
    5. token embedding과 context vector를 concat한다.
    6. GRU를 통해 다음 hidden state를 계산한다.
    7. linear layer를 통해 다음 token logits를 출력한다.

    이 구조는 source sentence를 하나의 고정 길이 vector에만 의존하지 않는다는 점에서
    기존 vanilla encoder-decoder보다 정보 병목을 완화한다.
    """

    def __init__(self, output_vocab_size, hidden_size, pad_idx=PAD_TOKEN):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, hidden_size, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_vocab_size)

    def forward_step(self, decoder_input, decoder_hidden, encoder_outputs, input_mask):
        embedded = self.embedding(decoder_input)
        query = decoder_hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)
        gru_input = torch.cat([embedded, context], dim=-1)
        output, hidden = self.gru(gru_input, decoder_hidden)
        logits = self.out(output)
        return logits, hidden, attn_weights

    def forward(self, encoder_outputs, encoder_hidden, input_mask, target_ids=None, max_len=MAX_LENGTH):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        decoder_input = torch.full((batch_size, 1), SOS_TOKEN, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden
        logits_all = []
        attn_all = []

        for t in range(max_len):
            logits, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, input_mask
            )
            logits_all.append(logits)
            attn_all.append(attn_weights.unsqueeze(1))
            if target_ids is not None:
                decoder_input = target_ids[:, t].unsqueeze(1)
            else:
                decoder_input = logits.argmax(dim=-1)

        logits_all = torch.cat(logits_all, dim=1)
        attn_all = torch.cat(attn_all, dim=1)
        return logits_all, decoder_hidden, attn_all


class EncoderDecoder(nn.Module):
    """
    Encoder와 Bahdanau Attention Decoder를 결합한 전체 seq2seq 모델이다.
    """

    def __init__(self, input_vocab_size, output_vocab_size, hidden_size):
        super().__init__()
        self.encoder = EncoderRNN(input_vocab_size, hidden_size)
        self.decoder = AttnDecoderRNN(output_vocab_size, hidden_size)

    def forward(self, input_ids, input_mask, target_ids=None):
        encoder_outputs, encoder_hidden = self.encoder(input_ids)
        logits, decoder_hidden, attn_weights = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, target_ids=target_ids
        )
        return logits, decoder_hidden, attn_weights
