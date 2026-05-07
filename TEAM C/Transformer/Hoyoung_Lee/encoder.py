"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        # 1. 임베딩 계층 (단어 임베딩 + 위치 인코딩(Positional Encoding))
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        # 2. N개의 인코더 레이어를 쌓음 (Multi-Head Attention + Feed Forward Network)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        # 입력된 단어 인덱스를 임베딩 벡터로 변환 (위치 정보 포함)
        x = self.emb(x)

        # 각 인코더 레이어를 순차적으로 통과
        for layer in self.layers:
            # src_mask: 패딩(Padding)된 부분을 어텐션 연산에서 무시하도록 하는 마스크
            x = layer(x, src_mask)

        # 최종 인코딩된 문맥 벡터 반환
        return x