from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer (한 층)

    구조:
    1. Self-Attention
    2. Add & Norm (Residual)
    3. Feed Forward Network
    4. Add & Norm (Residual)

    이 구조가 n_layers 만큼 반복됨 (보통 6층)
    """

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()

        # Multi-head self-attention
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)

        # 첫 번째 LayerNorm
        self.norm1 = LayerNorm(d_model=d_model)

        # dropout
        self.dropout1 = nn.Dropout(p=drop_prob)

        # Feed Forward Network
        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            hidden=ffn_hidden,
            drop_prob=drop_prob
        )

        # 두 번째 LayerNorm
        self.norm2 = LayerNorm(d_model=d_model)

        # dropout
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        """
        x: [batch, length, d_model]
        src_mask: padding mask
        """

        # ---------------------------------------------------
        # 1. Self-Attention
        # ---------------------------------------------------
        _x = x  # residual connection을 위해 저장

        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        # Q = K = V → self-attention
        # 문장 내부 단어 간 관계 학습

        # ---------------------------------------------------
        # 2. Add & Norm (Residual)
        # ---------------------------------------------------
        x = self.dropout1(x)

        x = self.norm1(x + _x)
        # residual connection
        # x + original input

        # ---------------------------------------------------
        # 3. Feed Forward Network
        # ---------------------------------------------------
        _x = x  # 다시 residual 위해 저장

        x = self.ffn(x)
        # 각 토큰별로 독립적인 MLP 처리

        # ---------------------------------------------------
        # 4. Add & Norm (Residual)
        # ---------------------------------------------------
        x = self.dropout2(x)

        x = self.norm2(x + _x)

        return x