"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

#단어 인덱스를 d_model 차원 벡터로 변환
class TokenEmbedding(nn.Embedding):
    """Token Embedding using torch.nn"""

    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
