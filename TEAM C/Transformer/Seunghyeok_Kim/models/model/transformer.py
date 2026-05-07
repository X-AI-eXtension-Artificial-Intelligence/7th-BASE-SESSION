import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder

# Transformer 클래스 구현
class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):      # Transformer 클래스의 생성자 메서드 __init__을 정의하여 모델의 하이퍼파라미터와 필요한 인스턴스들을 초기화
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)       # Encoder 클래스의 인스턴스를 생성하여 self.encoder에 할당하여 인코더에 사용

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)       # Decoder 클래스의 인스턴스를 생성하여 self.decoder에 할당하여 디코더에 사용

    def forward(self, src, trg):        # Transformer 클래스의 순전파 메서드 forward를 정의하여 모델의 입력(src, trg)을 받아 인코더와 디코더를 통해 출력(output)을 계산하여 반환
        src_mask = self.make_src_mask(src)      # 소스 마스크 생성 : 입력(src)에서 패딩 토큰의 위치를 마스킹하여 src_mask를 생성하여 반환
        trg_mask = self.make_trg_mask(trg)      # 타겟 마스크 생성 : 입력(trg)에서 패딩 토큰의 위치를 마스킹하고, 미래 토큰을 마스킹하여 trg_mask를 생성하여 반환
        enc_src = self.encoder(src, src_mask)                       # 인코더 계산 : 인코더에 입력(src)과 소스 마스크(src_mask)를 입력하여 인코더 계산을 수행하여 enc_src에 저장
        output = self.decoder(trg, enc_src, trg_mask, src_mask)     # 디코더 계산 : 디코더에 입력(trg), 인코더 출력(enc_src), 타겟 마스크(trg_mask), 소스 마스크(src_mask)를 입력하여 디코더 계산을 수행하여 output에 저장
        return output

    def make_src_mask(self, src):       # 소스 마스크 생성 : 입력(src)에서 패딩 토큰의 위치를 마스킹하여 src_mask를 생성하여 반환
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # 입력(src)에서 패딩 토큰의 위치를 마스킹하여 src_mask를 생성
        return src_mask

    def make_trg_mask(self, trg):       # 타겟 마스크 생성 : 입력(trg)에서 패딩 토큰의 위치를 마스킹하고, 미래 토큰을 마스킹하여 trg_mask를 생성하여 반환
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)      # 입력(trg)에서 패딩 토큰의 위치를 마스킹하여 trg_pad_mask를 생성
        trg_len = trg.shape[1]                                                  # 입력(trg)의 길이를 trg_len에 저장
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)      # 미래 토큰을 마스킹하여 trg_sub_mask를 생성 : trg_len x trg_len 크기의 하삼각 행렬을 생성하여 미래 토큰을 마스킹
        trg_mask = trg_pad_mask & trg_sub_mask      # trg_pad_mask와 trg_sub_mask를 논리곱(&) 연산하여 trg_mask를 생성 : 패딩 토큰과 미래 토큰을 모두 마스킹하여 최종 타겟 마스크(trg_mask)를 생성
        return trg_mask