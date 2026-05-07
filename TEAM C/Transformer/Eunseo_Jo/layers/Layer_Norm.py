import torch.nn as nn

import torch
#Add & Norm 부분의 Norm 정규화 부분
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm).__init__()
        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zeros(d_model))
        self.eps=eps

    def forward(self, x):
        mean=x.mean(-1,keepdim=True)
        var= x.var(-1,unbiased=False,keepdim=True)

        #가우시안 정규분포 만들어주기
        out=(x-mean)/torch.sqrt(var+self.eps) #math.sqrt 를 사용하면 안된다(스칼라용임)
        #왜 다시 기껏 정규화 해놨는데 평균과 분산을 이동 시키지?
        #원래 학습하면서 평균과 분산을 다르게 잘 학습하였으나, 너무 폭주해버리면 학습이 잘안되니까
        #정규화를 시켜야했으나 단점이 그러면 기껏 학습한 특징이 소실됨
        # 적당히 정규화하면서 , 적당히 이동해주는 방법을 택한것!

        #내적이 아니라 요소별 곱 주의
        out=self.gamma*out+self.beta  
        return out