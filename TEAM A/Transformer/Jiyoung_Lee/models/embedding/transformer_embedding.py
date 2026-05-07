from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding 합치는 클래스
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()

        # 단어 embedding
        self.tok_emb = TokenEmbedding(vocab_size, d_model)

        # 위치 encoding
        self.pos_emb = PositionalEncoding(d_model, max_len, device)

        # dropout
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        x: [batch, length]
        """

        # 1. 단어 embedding
        # shape: [batch, length, d_model]
        tok_emb = self.tok_emb(x)

        # 2. 위치 encoding
        # shape: [length, d_model]
        pos_emb = self.pos_emb(x)

        # 3. 둘을 더함 (broadcasting 발생)
        # → [batch, length, d_model]
        out = tok_emb + pos_emb

        # 4. dropout 적용
        out = self.drop_out(out)

        return out