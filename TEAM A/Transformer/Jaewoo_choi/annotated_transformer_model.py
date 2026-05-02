# annotated_transformer_model.py
#
# hyunwoongko/transformer 예제 구조를 기반으로 정리한 Transformer 실습 코드다.
#
# 핵심 참고:
# - Vaswani et al., 2017, Attention Is All You Need
# - hyunwoongko/transformer
#
# 이 파일은 모델 구조 설명용이다.
# 전체 학습 파이프라인은 Colab notebook/cell에서 수행했다.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    # Transformer는 RNN처럼 token을 순차적으로 처리하지 않기 때문에
    # token embedding에 위치 정보가 들어있지 않다.
    # 따라서 sin/cos 기반 positional encoding을 더해 순서 정보를 주입한다.

    def __init__(self, d_model, max_len, drop_prob):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


class ScaledDotProductAttention(nn.Module):
    # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    #
    # Q와 K의 dot-product를 통해 token 간 유사도를 계산한다.
    # sqrt(d_k)로 나누는 것은 값의 scale이 커져 softmax가 포화되는 것을 막기 위한 안정화 장치다.

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, v)

        return out, attn


class MultiHeadAttention(nn.Module):
    # Multi-head attention은 하나의 attention만 수행하지 않고,
    # 여러 head에서 서로 다른 projection을 통해 병렬 attention을 수행한다.
    #
    # 이를 통해 한 문장 안의 다양한 관계를 여러 관점에서 학습할 수 있다.

    def __init__(self, d_model, n_head):
        super().__init__()

        assert d_model % n_head == 0

        self.n_head = n_head
        self.d_tensor = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.w_concat = nn.Linear(d_model, d_model)

    def split(self, x):
        batch_size, length, d_model = x.size()
        x = x.view(batch_size, length, self.n_head, self.d_tensor)
        return x.transpose(1, 2)

    def concat(self, x):
        batch_size, head, length, d_tensor = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, length, head * d_tensor)

    def forward(self, q, k, v, mask=None):
        q = self.split(self.w_q(q))
        k = self.split(self.w_k(k))
        v = self.split(self.w_v(v))

        out, attn = self.attention(q, k, v, mask)

        out = self.concat(out)
        out = self.w_concat(out)

        return out, attn


class PositionwiseFeedForward(nn.Module):
    # 각 token 위치에 동일하게 적용되는 MLP다.
    # attention이 token 간 정보를 섞는다면,
    # FFN은 각 token 표현을 비선형적으로 변환한다.

    def __init__(self, d_model, ffn_hidden, drop_prob):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, d_model)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    # Encoder layer는 self-attention과 feed-forward network로 구성된다.
    # 각 sublayer 뒤에는 residual connection과 layer normalization이 적용된다.

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, src_mask):
        residual = x
        x, attn = self.self_attention(x, x, x, src_mask)
        x = self.norm1(residual + self.dropout1(x))

        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + self.dropout2(x))

        return x, attn


class DecoderLayer(nn.Module):
    # Decoder layer는 세 가지 sublayer를 가진다.
    # 1. masked self-attention
    # 2. encoder-decoder cross-attention
    # 3. position-wise feed-forward network
    #
    # masked self-attention은 미래 target token을 참조하지 못하게 한다.

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        residual = dec
        dec, self_attn = self.self_attention(dec, dec, dec, trg_mask)
        dec = self.norm1(residual + self.dropout1(dec))

        residual = dec
        dec, cross_attn = self.cross_attention(dec, enc, enc, src_mask)
        dec = self.norm2(residual + self.dropout2(dec))

        residual = dec
        dec = self.ffn(dec)
        dec = self.norm3(residual + self.dropout3(dec))

        return dec, self_attn, cross_attn
