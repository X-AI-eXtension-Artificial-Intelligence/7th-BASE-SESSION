import torch
from torch import nn

# 포지셔널 인코딩 클래스 정의
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, device):       # d_model, max_len, device를 입력으로 받는 초기화 메서드 정의
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)    # max_len과 d_model 크기의 0으로 채워진 텐서를 생성하여 self.encoding에 저장
        self.encoding.requires_grad = False                             # self.encoding 텐서의 requires_grad 속성을 False로 설정하여 학습 중에 업데이트되지 않도록 함

        pos = torch.arange(0, max_len, device=device)       # max_len 크기의 0부터 max_len-1까지의 정수로 이루어진 텐서를 생성하여 pos에 저장
        pos = pos.float().unsqueeze(dim=1)                  # pos 텐서를 float 타입으로 변환하고 unsqueeze를 사용하여 차원을 확장하여 단어의 위치를 나타내는 2D 텐서로 만듦

        _2i = torch.arange(0, d_model, step=2, device=device).float()       # _2i 텐서는 0부터 d_model-1까지의 정수 중에서 step이 2인 정수로 이루어진 텐서를 생성하여 단어 임베딩의 차원 인덱스를 나타냄

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))        # self.encoding 텐서의 짝수 인덱스(0, 2, 4, ...)에 위치 인코딩 값을 계산하여 저장. 위치 인코딩 값은 pos를 10000의 (_2i / d_model) 제곱으로 나눈 후에 sin 함수를 적용하여 계산됨
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))        # self.encoding 텐서의 홀수 인덱스(1, 3, 5, ...)에 위치 인코딩 값을 계산하여 저장. 위치 인코딩 값은 pos를 10000의 (_2i / d_model) 제곱으로 나눈 후에 cos 함수를 적용하여 계산됨

    def forward(self, x):                       # 입력 텐서 x를 받아서 포지셔널 인코딩을 적용하는 forward 메서드 정의

        batch_size, seq_len = x.size()          # 입력 텐서 x의 크기를 batch_size와 seq_len로 언패킹하여 저장. batch_size는 입력 시퀀스의 배치 크기, seq_len은 입력 시퀀스의 길이를 나타냄

        return self.encoding[:seq_len, :]       # self.encoding 텐서에서 seq_len 길이만큼의 포지셔널 인코딩을 반환. 반환되는 텐서의 크기는 (seq_len, d_model)이며, 입력 시퀀스의 길이에 맞게 포지셔널 인코딩이 적용됨
