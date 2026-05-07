"""
models/model/transformer.py
- Encoder와 Decoder를 합친 전체 Transformer 모델입니다.
"""

import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_pad_idx,
        trg_pad_idx,
        trg_sos_idx,
        enc_voc_size,
        dec_voc_size,
        d_model,
        n_head,
        max_len,
        ffn_hidden,
        n_layers,
        drop_prob,
        device
    ):
        super().__init__()

        # padding index와 시작 token index를 저장합니다.
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx

        # mask 생성 시 torch tensor를 올릴 장치입니다.
        self.device = device

        # Encoder: source 문장을 문맥 표현으로 바꿉니다.
        self.encoder = Encoder(
            enc_voc_size=enc_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            n_layers=n_layers,
            drop_prob=drop_prob,
            device=device
        )

        # Decoder: target prefix와 encoder 출력을 사용해 다음 단어를 예측합니다.
        self.decoder = Decoder(
            dec_voc_size=dec_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            n_layers=n_layers,
            drop_prob=drop_prob,
            device=device
        )

    def make_src_mask(self, src):
        """source padding token을 attention에서 제외하기 위한 mask입니다."""
        # src shape: [batch, src_len]
        # mask shape: [batch, 1, 1, src_len]
        # head 차원과 query length 차원으로 broadcasting됩니다.
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        """target padding + 미래 token을 가리는 mask입니다."""
        # padding mask: [batch, 1, 1, trg_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        trg_len = trg.shape[1]

        # no-peak mask: [trg_len, trg_len]
        # 아래삼각행렬을 만들어 현재 위치가 미래 위치를 보지 못하게 합니다.
        trg_sub_mask = torch.tril(
            torch.ones(trg_len, trg_len)
        ).type(torch.ByteTensor).to(self.device)

        # padding mask와 미래 가림 mask를 동시에 적용합니다.
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, src, trg):
        # src shape: [batch, src_len]
        # trg shape: [batch, trg_len]

        # 1. mask 생성
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # 2. source 문장을 Encoder에 통과시킵니다.
        enc_src = self.encoder(src, src_mask)

        # 3. target prefix와 encoder memory를 Decoder에 넣습니다.
        output = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output shape: [batch, trg_len, dec_voc_size]
        return output
