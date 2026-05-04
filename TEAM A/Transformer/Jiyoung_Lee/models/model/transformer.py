import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()

        # padding index 저장 (mask 생성에 사용)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        # decoder 시작 토큰
        self.trg_sos_idx = trg_sos_idx

        self.device = device

        # encoder 생성
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        # decoder 생성
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        # src_mask: padding 위치를 attention에서 제외하기 위한 mask
        src_mask = self.make_src_mask(src)

        # trg_mask: padding + 미래 단어 차단 mask
        trg_mask = self.make_trg_mask(trg)

        # encoder 출력 (입력 문장 전체 representation)
        enc_src = self.encoder(src, src_mask)

        # decoder 실행 (입력 문장 + 이전 출력 토큰 사용)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output

    def make_src_mask(self, src):
        # src != pad 인 위치는 1, pad는 0 -> 특정 pad 열만 못보게 함 
        # shape: [batch, 1, 1, src_len]

        # 원래 src shaper: [batch, src_len] 
        # -> unsqueeze(1): [batch, 1, src_len] 
        # -> unsqueeze(2): [batch, 1, 1, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        # padding mask 
        # shape: [batch, 1, trg_len, 1]

        # trg 원래 shape: [batch, trg_len]
        # -> unsqueeze(1): [batch, 1, trg_len]
        # -> unsqueeze(3): [batch, 1, trg_len, 1]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        trg_len = trg.shape[1]

        # 미래 단어를 못 보게 하는 mask (lower triangular)
        # shape: [trg_len, trg_len]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)

        # 두 mask를 결합
        # shape: [batch, 1, trg_len, trg_len]
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask