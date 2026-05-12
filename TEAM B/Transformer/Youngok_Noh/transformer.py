"""
models/layers/position_wise_feed_forward.py
- Transformer block 안의 FFN입니다.
- 각 token 위치마다 같은 MLP를 독립적으로 적용합니다.
"""

from torch import nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()

        # 첫 번째 선형층: d_model → hidden
        self.linear1 = nn.Linear(d_model, hidden)

        # 비선형 활성화 함수입니다.
        self.relu = nn.ReLU()

        # dropout으로 과적합을 줄입니다.
        self.dropout = nn.Dropout(p=drop_prob)

        # 두 번째 선형층: hidden → d_model
        # 출력 차원을 다시 residual connection과 더할 수 있는 크기로 맞춥니다.
        self.linear2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
