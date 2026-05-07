"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        # 1. 타겟 언어용 임베딩 계층 (단어 임베딩 + 위치 인코딩)
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        # 2. N개의 디코더 레이어를 쌓음 (Masked Attention + Encoder-Decoder Attention + FFN)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        # 3. 최종 출력 계층: 디코더의 출력을 타겟 어휘 사전 크기의 확률값으로 변환 (LM Head)
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # 타겟 문장 임베딩
        trg = self.emb(trg)

        # 각 디코더 레이어를 순차적으로 통과
        for layer in self.layers:
            # enc_src: 인코더의 최종 출력 (Encoder-Decoder Attention에 사용)
            # trg_mask: 미래 단어 참조 방지(Look-ahead) 및 패딩 마스크
            # src_mask: 인코더 출력 중 패딩된 부분 무시
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # 선형 계층(LM head)을 통과시켜 다음 단어에 대한 로짓(Logits) 반환
        output = self.linear(trg)
        return output