"""
models/layers/scale_dot_product_attention.py
- Scaled Dot-Product Attention을 구현합니다.
"""

import math

import torch
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # q, k, v shape: [batch, head, seq_len, d_tensor]
        # batch_size, head, length, d_tensor 순서입니다.
        batch_size, head, length, d_tensor = k.size()

        # 1. QK^T를 계산합니다.
        # k_t shape: [batch, head, d_tensor, seq_len]
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 2. mask가 있으면 attention score에서 해당 위치를 매우 작은 값으로 바꿉니다.
        # softmax 이후 거의 0이 되도록 -10000을 넣습니다.
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. softmax로 attention distribution을 만듭니다.
        score = self.softmax(score)

        # 4. attention distribution을 value에 곱합니다.
        # 결과 shape: [batch, head, seq_len, d_tensor]
        v = score @ v

        return v, score
