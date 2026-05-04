from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    """
    Transformer의 인코더(Encoder) 모듈.
    소스 시퀀스를 입력받아 각 토큰의 문맥 정보가 풍부하게 담긴
    연속 표현(continuous representation)으로 변환한다.

    구조:
        1. TransformerEmbedding  : 토큰 임베딩 + Positional Encoding
        2. EncoderLayer x N      : Self-Attention → FFN 블록을 N번 반복
    """

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        # 소스 토큰 인덱스를 d_model 차원의 벡터로 변환 + Positional Encoding 추가
        # Positional Encoding은 순서 정보가 없는 어텐션 연산에 위치 정보를 주입하는 역할
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        # EncoderLayer를 n_layers개 쌓아서 ModuleList로 관리
        # 각 레이어는 다음 두 단계로 구성됨:
        #   1) Multi-Head Self-Attention : 시퀀스 내 모든 토큰 간의 관계(문맥)를 포착
        #   2) Feed-Forward Network      : 위치별 독립적인 비선형 변환으로 표현력 강화
        # 레이어를 깊이 쌓을수록 더 추상적이고 풍부한 문맥 표현을 학습함
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        """
        인코더 순전파

        입력:
            x        : 소스 시퀀스 토큰 인덱스. shape = (batch_size, src_len)
            src_mask : 소스 패딩 마스크. shape = (batch_size, 1, 1, src_len)
                       Self-Attention 계산 시 패딩 토큰의 영향을 제거하기 위해 사용됨.

        출력:
            x : 인코더의 최종 출력 (문맥 벡터). shape = (batch_size, src_len, d_model)
                디코더의 Cross-Attention에서 Key, Value로 활용됨.
        """
        # 1단계: 소스 토큰 인덱스 → 임베딩 벡터 + Positional Encoding
        # shape: (batch_size, src_len) → (batch_size, src_len, d_model)
        x = self.emb(x)

        # 2단계: N개의 EncoderLayer를 순차적으로 통과
        # 각 레이어를 거칠수록 토큰 표현에 점점 더 풍부한 문맥 정보가 누적됨
        for layer in self.layers:
            x = layer(x, src_mask)

        # 최종 출력: 디코더의 Cross-Attention에서 enc_src로 사용됨
        return x