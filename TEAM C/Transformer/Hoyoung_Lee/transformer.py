"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        # 패딩(Padding) 및 시작(SOS) 토큰 인덱스 저장
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        
        # 인코더와 디코더 객체 초기화
        self.encoder = Encoder(d_model=d_model, n_head=n_head, max_len=max_len,
                               ffn_hidden=ffn_hidden, enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob, n_layers=n_layers, device=device)

        self.decoder = Decoder(d_model=d_model, n_head=n_head, max_len=max_len,
                               ffn_hidden=ffn_hidden, dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob, n_layers=n_layers, device=device)

    def forward(self, src, trg):
        # 1. 소스와 타겟 문장에 대한 마스크 생성
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        # 2. 소스 문장을 인코더에 통과
        enc_src = self.encoder(src, src_mask)
        
        # 3. 타겟 문장과 인코더 출력을 디코더에 통과
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    # --- 마스크 생성 함수들 ---
    def make_src_mask(self, src):
        # 패딩 토큰(src_pad_idx)인 부분은 0(False), 의미 있는 단어는 1(True)로 마스킹
        # 형태: [batch_size, 1, 1, src_len] (Multi-Head Attention 연산을 위해 차원 추가)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        # 1. 패딩 마스크 (타겟 문장의 패딩 토큰 가림)
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        
        # 2. 미래 정보 마스크 (Look-ahead mask)
        # 디코더는 단어를 순차적으로 예측해야 하므로, 현재 타임스텝보다 뒤에 있는 미래 단어를 보지 못하게 하삼각행렬(tril)로 마스킹
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        
        # 패딩 마스크와 미래 정보 마스크를 AND(&) 연산으로 결합
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask