from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):

    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        # decoder 입력 embedding
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        # decoder layer 여러 개
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model,
                         ffn_hidden=ffn_hidden,
                         n_head=n_head,
                         drop_prob=drop_prob)
            for _ in range(n_layers)
        ])

        # 최종 단어 예측용 linear layer
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        trg: [batch, trg_len]
        enc_src: [batch, src_len, d_model]
        trg_mask: [batch, 1, trg_len, trg_len]
        src_mask: [batch, 1, 1, src_len]
        """

        # 1. embedding
        # [batch, trg_len] → [batch, trg_len, d_model]
        trg = self.emb(trg)

        # 2. decoder layer 반복
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
            # shape 유지: [batch, trg_len, d_model]

        # 3. linear → vocab 확률로 변환
        output = self.linear(trg)
        # [batch, trg_len, vocab_size]

        return output