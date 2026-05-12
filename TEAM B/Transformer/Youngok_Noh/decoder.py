"""
models/layers/layer_norm.py
- Layer Normalization을 직접 구현한 파일입니다.
"""

import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()

        # gamma: 정규화 후 scale을 조정하는 학습 파라미터입니다.
        self.gamma = nn.Parameter(torch.ones(d_model))

        # beta: 정규화 후 shift를 조정하는 학습 파라미터입니다.
        self.beta = nn.Parameter(torch.zeros(d_model))

        # 분산이 0에 가까울 때 나눗셈이 불안정해지는 것을 막습니다.
        self.eps = eps

    def forward(self, x):
        # 마지막 차원(d_model)에 대해 평균과 분산을 구합니다.
        # shape 유지를 위해 keepdim=True를 씁니다.
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        # 표준화: 평균 0, 분산 1에 가깝게 만듭니다.
        out = (x - mean) / torch.sqrt(var + self.eps)

        # 학습 가능한 gamma/beta로 다시 scale/shift합니다.
        out = self.gamma * out + self.beta

        return out
