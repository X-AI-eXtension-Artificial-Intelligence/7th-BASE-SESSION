"""
models/blocks/encoder_layer.py
- Transformer Encoder의 한 층을 구현합니다.
"""

from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()

        # Encoder self-attention입니다.
        # q, k, v가 모두 encoder 입력 x에서 나옵니다.
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)

        # Add & Norm의 Norm 부분입니다.
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # Position-wise FFN입니다.
        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            hidden=ffn_hidden,
            drop_prob=drop_prob
        )

        # 두 번째 Add & Norm입니다.
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # x shape: [batch, src_len, d_model]

        # residual connection을 위해 원본 x를 저장합니다.
        _x = x

        # 1. Self-Attention
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. Dropout + Residual + LayerNorm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # FFN에도 residual connection을 적용합니다.
        _x = x

        # 3. Position-wise Feed Forward
        x = self.ffn(x)

        # 4. Dropout + Residual + LayerNorm
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x
