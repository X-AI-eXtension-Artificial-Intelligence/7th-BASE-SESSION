"""
models/blocks/decoder_layer.py
- Transformer Decoder의 한 층을 구현합니다.
"""

from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()

        # 1. target 문장 내부를 보는 masked self-attention입니다.
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # 2. Encoder 출력값을 참고하는 encoder-decoder attention입니다.
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        # 3. Position-wise FFN입니다.
        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            hidden=ffn_hidden,
            drop_prob=drop_prob
        )
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # dec shape: [batch, trg_len, d_model]
        # enc shape: [batch, src_len, d_model]

        # 1. Masked self-attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 2. Encoder-Decoder Attention
        # enc가 None이 아닐 때 source 문장 정보를 참조합니다.
        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 3. Position-wise FFN
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)

        return x
