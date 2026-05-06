"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

#평균·분산 정규화 후 γ, β 파라미터로 스케일 복원. 
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.eps   = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var  = x.var(-1, unbiased=False, keepdim=True)
        out  = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * out + self.beta
