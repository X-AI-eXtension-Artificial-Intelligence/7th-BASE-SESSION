import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

# Decoder 클래스 구현
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)              # TransformerEmbedding 클래스의 인스턴스를 생성하여 self.emb에 할당하여 디코더 임베딩에 사용

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])     # DecoderLayer 클래스의 인스턴스를 n_layers 개수만큼 생성하여 self.layers에 할당하여 디코더 레이어에 사용

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)     # 디코더 임베딩 계산 : 디코더의 입력 텐서(trg)를 self.emb에 입력하여 디코더 임베딩을 계산하여 trg에 저장

        for layer in self.layers:                           # 디코더 레이어 반복 : self.layers에 저장된 각 디코더 레이어(layer)에 대해 반복하여 디코더 레이어 계산을 수행
            trg = layer(trg, enc_src, trg_mask, src_mask)   # 디코더 레이어 계산 : 현재 레이어(layer)에 trg, enc_src, trg_mask, src_mask를 입력하여 디코더 레이어 계산을 수행하고, 결과를 trg에 저장

        output = self.linear(trg)       # 선형 변환 : 디코더 레이어 계산 결과(trg)를 self.linear에 입력하여 선형 변환을 수행하여 output에 저장
        return output