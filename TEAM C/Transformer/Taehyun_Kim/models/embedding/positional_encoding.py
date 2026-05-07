"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

#sin/cos 함수로 위치 정보 행렬 계산
class PositionalEncoding(nn.Module):
    """compute sinusoid encoding."""

    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos  = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        _2i  = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        # x: [batch_size, seq_len]
        seq_len = x.size(1)
        return self.encoding[:seq_len, :]
