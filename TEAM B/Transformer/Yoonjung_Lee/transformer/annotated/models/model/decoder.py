"""
models/model/decoder.py
- embedding + 여러 DecoderLayer + 최종 단어 분류 linear layer로 Decoder를 구성합니다.
"""

from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(
        self,
        dec_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers,
        drop_prob,
        device
    ):
        super().__init__()

        # target token index를 Transformer 입력 벡터로 바꿉니다.
        self.emb = TransformerEmbedding(
            d_model=d_model,
            drop_prob=drop_prob,
            max_len=max_len,
            vocab_size=dec_voc_size,
            device=device
        )

        # DecoderLayer를 n_layers개 쌓습니다.
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                drop_prob=drop_prob
            )
            for _ in range(n_layers)
        ])

        # 각 위치의 hidden state를 target vocabulary 크기의 logit으로 바꿉니다.
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg shape: [batch, trg_len]
        # enc_src shape: [batch, src_len, d_model]

        # target token index → embedding vector
        trg = self.emb(trg)

        # 각 DecoderLayer를 순서대로 통과합니다.
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # vocabulary별 점수로 변환합니다.
        output = self.linear(trg)

        return output
