from torch import nn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()

        # d_model → hidden (확장) 
        # (ex. [batch, length, 512] → [batch, length, 2048])
        self.linear1 = nn.Linear(d_model, hidden)

        # hidden → d_model (다시 축소)
        # (ex. [batch, length, 2048] → [batch, length, 512])
        self.linear2 = nn.Linear(hidden, d_model)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # x: [batch, length, d_model]

        x = self.linear1(x)
        # [batch, length, hidden]

        x = self.relu(x)

        x = self.dropout(x)

        x = self.linear2(x)
        # [batch, length, d_model]

        return x