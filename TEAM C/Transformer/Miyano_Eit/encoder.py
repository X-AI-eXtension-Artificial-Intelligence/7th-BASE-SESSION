import math
import torch.nn as nn
from model_parts.attention import MultiHeadAttention
from model_parts.layers import PositionwiseFeedForward, PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, h)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_output, _ = self.self_attention(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, h, d_ff, N, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)]
        )
        self.scale = math.sqrt(d_model)

    def forward(self, src, src_mask=None):
        x = self.embedding(src) * self.scale
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x