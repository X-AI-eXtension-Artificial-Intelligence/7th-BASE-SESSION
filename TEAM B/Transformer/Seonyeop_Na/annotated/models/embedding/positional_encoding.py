"""
models/embedding/positional_encoding.py
- Transformer에 순서 정보를 넣기 위한 sinusoidal positional encoding입니다.
"""

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        # encoding shape: [max_len, d_model]
        # 각 위치 pos마다 d_model 차원의 위치 벡터를 미리 계산해 둡니다.
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        # pos shape: [max_len, 1]
        # 문장 내 위치 번호입니다. 0, 1, 2, ..., max_len-1
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        # _2i shape: [d_model/2]
        # 짝수/홀수 차원에 sin/cos를 번갈아 넣기 위한 index입니다.
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 짝수 차원에는 sin, 홀수 차원에는 cos를 넣습니다.
        # 이렇게 하면 모델이 상대적 위치 관계를 학습하기 쉬워집니다.
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        batch_size, seq_len = x.size()

        # 필요한 문장 길이만큼만 잘라서 반환합니다.
        # 반환 shape: [seq_len, d_model]
        # batch 차원은 broadcasting으로 더해집니다.
        return self.encoding[:seq_len, :]
