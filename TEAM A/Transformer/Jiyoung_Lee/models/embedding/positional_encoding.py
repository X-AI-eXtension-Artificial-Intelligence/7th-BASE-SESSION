import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    위치 정보를 sinusoidal 함수로 생성
    """

    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        # 위치 encoding 저장 공간
        # shape: [max_len, d_model]
        self.encoding = torch.zeros(max_len, d_model, device=device)

        # gradient 계산 안 함 (고정된 값)
        self.encoding.requires_grad = False

        # 위치 index 생성
        # shape: [max_len]
        pos = torch.arange(0, max_len, device=device).float()

        # shape: [max_len, 1]
        pos = pos.unsqueeze(dim=1)

        # d_model 차원 index (짝수 위치)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 짝수 index → sin
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))

        # 홀수 index → cos
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        """
        x: [batch, length]
        """

        batch_size, seq_len = x.size()

        # 앞에서부터 seq_len만큼 잘라서 반환
        # shape: [seq_len, d_model]
        return self.encoding[:seq_len, :]