import math
import torch.nn as nn;
import torch.nn.functional as F

class ScaleDotProductAttention(nn.Module):
    def __init__(self, ):
        super(ScaleDotProductAttention).__init__()
        
    def forward(self, q,k,v,mask=None,e=1e-12):
        #차원 확인
        batch_size,head,length,d_model=k.size()

        #행렬곱을 위한 k행렬 transpose
        k_t=k.transpose(2,3)
        score=q@k_t / math.sqrt(d_model)

        #마스크를 통해서 음의 무한대로 보내기
        if mask is not None:
            score=score.masked_fill(mask==0, -10000)

        #반드시 mask적용을 한다음에 softmax를 해주어야함
        #만약 softmax를 적용하고 mask를 한다면 합이 1이 안될 수 있음
        score=F.softmax(score)
        V=score@v;

        #V행렬 뿐만아니라 어디에 유사도가 높게 나왔는지 확인하기 위해 score행렬을 return
        return V,score

        
