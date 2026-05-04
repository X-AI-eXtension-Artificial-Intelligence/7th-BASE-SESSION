import torch
from torch import nn

# 각 단어 벡터를 "평균 0, 분산 1"로 정규화 
# Ex. [0.3, -1.2, 5.1] → [-0.2, -1.0, 1.2]
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()

        # scale 파라미터 (학습됨)
        self.gamma = nn.Parameter(torch.ones(d_model))

        # shift 파라미터 (학습됨)
        self.beta = nn.Parameter(torch.zeros(d_model))

        # 0으로 나누는 것 방지
        self.eps = eps

    def forward(self, x):
        # x: [batch, length, d_model]

        # 마지막 차원 기준 평균
        mean = x.mean(-1, keepdim=True)

        # 마지막 차원 기준 분산
        var = x.var(-1, unbiased=False, keepdim=True)

        # 정규화
        out = (x - mean) / torch.sqrt(var + self.eps)

        # scale + shift
        out = self.gamma * out + self.beta

        return out