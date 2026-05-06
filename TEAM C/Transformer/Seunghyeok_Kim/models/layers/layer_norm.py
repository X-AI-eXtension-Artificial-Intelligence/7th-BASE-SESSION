import torch
from torch import nn

# 레이어 정규화 클래스 정의
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):                 # d_model과 eps를 입력으로 받는 초기화 메서드 정의
        super(LayerNorm, self).__init__()                   
        self.gamma = nn.Parameter(torch.ones(d_model))      # d_model 크기의 1로 채워진 텐서를 생성하여 self.gamma에 저장. self.gamma는 학습 가능한 파라미터로, 레이어 정규화 후에 스케일링을 적용하는 데 사용됨
        self.beta = nn.Parameter(torch.zeros(d_model))      # d_model 크기의 0으로 채워진 텐서를 생성하여 self.beta에 저장. self.beta는 학습 가능한 파라미터로, 레이어 정규화 후에 시프트를 적용하는 데 사용됨
        self.eps = eps                                      # 레이어 정규화에서 분모가 0이 되는 것을 방지하기 위한 작은 상수 eps를 self.eps에 저장

    def forward(self, x):                                   # 입력 텐서 x에 레이어 정규화를 적용하는 forward 메서드 정의
        mean = x.mean(-1, keepdim=True)                     # 입력 텐서 x의 마지막 차원에 대해 평균을 계산하여 mean에 저장. keepdim=True로 설정하여 mean의 차원이 유지되도록 함
        var = x.var(-1, unbiased=False, keepdim=True)       # 입력 텐서 x의 마지막 차원에 대해 분산을 계산하여 var에 저장. unbiased=False로 설정하여 편향 없는 분산을 계산하고, keepdim=True로 설정하여 var의 차원이 유지되도록 함

        out = (x - mean) / torch.sqrt(var + self.eps)       # 레이어 정규화를 적용하여 out에 저장. self.eps는 분모가 0이 되는 것을 방지하기 위한 작은 상수임
        out = self.gamma * out + self.beta                  # 레이어 정규화 후에 스케일링과 시프트를 적용하여 최종 출력을 계산함
        return out                                          # 최종 출력을 반환함
