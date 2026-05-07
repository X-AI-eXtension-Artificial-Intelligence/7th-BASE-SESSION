import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    """
    Transformer의 디코더(Decoder) 모듈.
    타겟 시퀀스를 입력받아 인코더의 출력(enc_src)과 결합하여
    각 위치에서 다음 토큰의 확률 분포를 예측한다.

    구조:
        1. TransformerEmbedding  : 토큰 임베딩 + Positional Encoding
        2. DecoderLayer x N      : Masked Self-Attention → Cross-Attention → FFN 블록을 N번 반복
        3. Linear (LM Head)      : d_model 차원 → dec_voc_size 차원으로 투영하여 각 토큰의 logit 생성
    """

    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        # 타겟 토큰 인덱스를 d_model 차원의 벡터로 변환 + Positional Encoding 추가
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        # DecoderLayer를 n_layers개 쌓아서 ModuleList로 관리
        # 각 레이어는 다음 세 단계로 구성됨:
        #   1) Masked Self-Attention  : look-ahead 마스크로 미래 토큰 참조 방지
        #   2) Cross-Attention        : 인코더 출력(enc_src)과의 어텐션으로 소스 문맥 반영
        #   3) Feed-Forward Network   : 위치별 독립적인 비선형 변환
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        # LM Head: 최종 hidden state를 어휘 크기의 logit으로 변환
        # shape 변환: (batch_size, trg_len, d_model) → (batch_size, trg_len, dec_voc_size)
        # 이후 softmax를 거쳐 각 위치에서의 토큰 확률 분포가 된다.
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        디코더 순전파

            trg      : 타겟 시퀀스 토큰 인덱스. shape = (batch_size, trg_len)
            enc_src  : 인코더의 최종 출력 (문맥 벡터). shape = (batch_size, src_len, d_model)
            trg_mask : 타겟 마스크 (패딩 + look-ahead). shape = (batch_size, 1, trg_len, trg_len)
            src_mask : 소스 패딩 마스크. shape = (batch_size, 1, 1, src_len)
                       Cross-Attention 시 소스 패딩 위치를 무시하기 위해 사용됨.

        """
        # 1단계: 타겟 토큰 인덱스 → 임베딩 벡터 + Positional Encoding
        # shape: (batch_size, trg_len) → (batch_size, trg_len, d_model)
        trg = self.emb(trg)

        # 2단계: N개의 DecoderLayer를 순차적으로 통과
        # 각 레이어마다 Masked Self-Attention → Cross-Attention → FFN 수행
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # 3단계: LM Head를 통해 최종 logit 생성
        # shape: (batch_size, trg_len, d_model) → (batch_size, trg_len, dec_voc_size)
        output = self.linear(trg)
        return output