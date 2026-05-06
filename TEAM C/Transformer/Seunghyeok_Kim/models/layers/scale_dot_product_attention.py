import math

from torch import nn

# Scaled Dot Product Attention 클래스 구현
class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)       # nn.Softmax는 소프트맥스 함수를 적용하는 레이어입니다. dim=-1로 설정하여 마지막 차원에 대해 소프트맥스 함수를 적용하도록 합니다.

    def forward(self, q, k, v, mask=None, e=1e-12):     # forward 메서드는 ScaleDotProductAttention 클래스의 인스턴스가 호출될 때 실행되는 메서드입니다.
        batch_size, head, length, d_tensor = k.size()

        # 1. calculate score
        k_t = k.transpose(2, 3)                         
        score = (q @ k_t) / math.sqrt(d_tensor)         

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. softmax
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score