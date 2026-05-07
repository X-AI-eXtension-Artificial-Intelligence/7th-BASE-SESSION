"""
models/model/encoder.py
- embedding + 여러 EncoderLayer를 쌓아 Encoder를 구성합니다.
"""

from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(
        self,
        enc_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers,
        drop_prob,
        device
    ):
        super().__init__()

        # source token index를 Transformer 입력 벡터로 바꿉니다.
        self.emb = TransformerEmbedding(
            d_model=d_model,
            max_len=max_len,
            vocab_size=enc_voc_size,
            drop_prob=drop_prob,
            device=device
        )

        # EncoderLayer를 n_layers개 쌓습니다.
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                drop_prob=drop_prob
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, src_mask):
        # x shape: [batch, src_len]

        # token index → embedding vector
        x = self.emb(x)

        # 각 EncoderLayer를 순서대로 통과합니다.
        for layer in self.layers:
            x = layer(x, src_mask)

        # 최종 encoder memory입니다.
        return x
