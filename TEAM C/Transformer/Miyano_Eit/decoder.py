import math
import torch.nn as nn
from model_parts.attention import MultiHeadAttention
from model_parts.layers import PositionwiseFeedForward, PositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, h)
        self.cross_attention = MultiHeadAttention(d_model, h)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attn_output, _ = self.masked_self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        attn_output, attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, h, d_ff, N, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)]
        )
        self.scale = math.sqrt(d_model)

    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt) * self.scale
        x = self.pos_encoding(x)

        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, encoder_output, src_mask, tgt_mask)
            attn_weights_list.append(attn_weights)

        return x, attn_weights_list