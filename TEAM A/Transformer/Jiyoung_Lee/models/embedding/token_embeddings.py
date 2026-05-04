from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    nn.Embedding을 그대로 상속받은 클래스

    역할:
    단어 index → 벡터로 변환
    """

    def __init__(self, vocab_size, d_model):
        """
        vocab_size: 단어 개수
        d_model: embedding 차원 (보통 512)
        """

        # padding_idx=1 → padding 토큰은 항상 0 벡터로 유지됨
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)