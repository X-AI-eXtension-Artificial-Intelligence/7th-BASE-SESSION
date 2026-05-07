from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()

        # 1. Masked Self-Attention
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # 2. Encoder-Decoder Attention
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        # 3. Feed Forward
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        """
        dec: decoder input (현재까지 생성된 단어들)
        enc: encoder output (입력 문장 정보)
        trg_mask: 미래 단어 차단 mask
        src_mask: padding mask
        """

        # ============================
        # 1. Masked Self-Attention
        # ============================

        _x = dec  # residual 저장

        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        # decoder 내부 attention
        # BUT mask 때문에 미래 단어 못 봄

        # ============================
        # 2. Add & Norm
        # ============================

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # ============================
        # 3. Encoder-Decoder Attention
        # ============================

        if enc is not None:
            _x = x  # residual 저장

            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            # Q = decoder
            # K,V = encoder
            # decoder가 input 문장을 참고하는 단계

            # ============================
            # 4. Add & Norm
            # ============================

            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # ============================
        # 5. Feed Forward Network
        # ============================

        _x = x
        x = self.ffn(x)
        # 각 단어 독립적으로 MLP 통과

        # ============================
        # 6. Add & Norm
        # ============================

        x = self.dropout3(x)
        x = self.norm3(x + _x)

        return x