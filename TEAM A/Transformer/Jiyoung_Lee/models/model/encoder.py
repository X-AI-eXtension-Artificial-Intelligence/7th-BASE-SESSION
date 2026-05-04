from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        # 입력 → embedding (token + positional)
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        # encoder layer 여러 개 쌓기 (보통 6개)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model,
                         ffn_hidden=ffn_hidden,
                         n_head=n_head,
                         drop_prob=drop_prob)
            for _ in range(n_layers)
        ])

    def forward(self, x, src_mask):
        """
        x: [batch, src_len]
        src_mask: [batch, 1, 1, src_len]
        """

        # 1. embedding
        # [batch, src_len] → [batch, src_len, d_model]
        x = self.emb(x)

        # 2. encoder layer 반복
        for layer in self.layers:
            x = layer(x, src_mask)
            # shape 유지: [batch, src_len, d_model]

        # 3. 최종 encoder output
        return x