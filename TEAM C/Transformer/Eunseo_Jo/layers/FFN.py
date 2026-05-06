import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model,hidden,drop_prob=0.1):
        super(PositionwiseFeedForward).__init__()
        self.layer1=nn.Linear(d_model,hidden)
        self.layer2=nn.Linear(hidden,d_model)
        self.relu=nn.ReLu()
        self.drop_out=nn.Dropout(p=drop_prob)

    def forward(self,x):
        #1단계 팽창 512 -> 2048
        x=self.layer1(x)
        #2단계 비선형 ReLu 통과
        x=self.relu(x)
        #3단계 dropOut (=> 과적합 방지)
        x=self.drop_out(x)
        #4단계 축소 2048 -> 512
        x=self.layer2(x)
        return x
        