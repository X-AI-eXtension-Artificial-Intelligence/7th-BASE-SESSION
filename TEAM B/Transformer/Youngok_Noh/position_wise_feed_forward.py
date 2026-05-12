"""
models/embedding/token_embeddings.py
- 단어 index를 dense vector로 바꾸는 embedding layer입니다.
"""

from torch import nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        # nn.Embedding을 상속합니다.
        # vocab_size: 단어장 크기
        # d_model: 각 토큰을 표현할 벡터 차원
        # padding_idx=1: <pad> 토큰 index가 1이라고 가정하고, 해당 vector는 학습 업데이트에서 제외합니다.
        super(TokenEmbedding, self).__init__(
            vocab_size,
            d_model,
            padding_idx=1
        )
