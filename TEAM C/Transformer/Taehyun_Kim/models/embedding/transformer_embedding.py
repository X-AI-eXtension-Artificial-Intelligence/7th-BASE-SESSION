"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn
from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding

#위 둘을 합산해서 드롭아웃 적용. 모든 입력이 여기를 거쳐서 encoder/decoder로 들어감.
class TransformerEmbedding(nn.Module):
    """token embedding + positional encoding (sinusoid)"""

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb  = TokenEmbedding(vocab_size, d_model)
        self.pos_emb  = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
