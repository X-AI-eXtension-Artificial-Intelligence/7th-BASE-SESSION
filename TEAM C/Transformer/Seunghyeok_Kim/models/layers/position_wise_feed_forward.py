from torch import nn

# Position-wise Feed Forward 클래스 구현
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):     # d_model : 모델의 차원, hidden : feed forward network의 hidden layer의 차원, drop_prob : dropout 확률
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)           # nn.Linear은 선형 변환을 수행하는 레이어입니다. 이 레이어는 feed forward network의 첫 번째 레이어로 사용됩니다.
        self.linear2 = nn.Linear(hidden, d_model)           # nn.Linear은 선형 변환을 수행하는 레이어입니다. 이 레이어는 feed forward network의 두 번째 레이어로 사용됩니다.
        self.relu = nn.ReLU()                               # nn.ReLU는 ReLU 활성화 함수를 적용하는 레이어입니다. self.relu에 할당하여 feed forward network의 활성화 함수로 사용됩니다.
        self.dropout = nn.Dropout(p=drop_prob)              # nn.Dropout은 드롭아웃을 적용하는 레이어입니다. 여기서는 드롭아웃 확률을 drop_prob로 설정하여 self.dropout에 할당합니다.

    def forward(self, x):       # forward 메서드는 PositionwiseFeedForward 클래스의 인스턴스가 호출될 때 실행되는 메서드입니다.
        x = self.linear1(x)     
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
