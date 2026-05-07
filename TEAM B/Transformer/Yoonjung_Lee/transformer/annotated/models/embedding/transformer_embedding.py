"""
models/embedding/transformer_embedding.py
- token embedding과 positional encoding을 더해서 Transformer 입력 벡터를 만듭니다.
"""

from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    최종 embedding = token embedding + positional encoding
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()

        # 단어 index를 d_model 차원의 벡터로 바꿉니다.
        self.tok_emb = TokenEmbedding(vocab_size, d_model)

        # 위치 정보를 나타내는 sinusoidal encoding입니다.
        self.pos_emb = PositionalEncoding(d_model, max_len, device)

        # embedding 단계에서도 dropout을 적용합니다.
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # token embedding: [batch, seq_len, d_model]
        # positional encoding: [seq_len, d_model]
        # broadcasting으로 두 텐서가 더해집니다.
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)

        # 단어 의미 + 위치 정보를 합친 뒤 dropout을 적용합니다.
        return self.drop_out(tok_emb + pos_emb)
