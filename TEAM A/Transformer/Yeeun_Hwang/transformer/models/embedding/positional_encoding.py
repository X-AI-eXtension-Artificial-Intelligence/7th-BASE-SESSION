import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding 구현.

    Transformer는 RNN과 달리 시퀀스를 순차적으로 처리하지 않기 때문에 토큰의 순서(위치) 정보를 별도 주입 필요.

    수식:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        - pos : 시퀀스에서 토큰의 위치 (0 ~ max_len-1)
        - i   : 임베딩 차원의 인덱스 (0 ~ d_model/2 - 1)

    짝수 차원(0, 2, 4, ...)에는 sin, 홀수 차원(1, 3, 5, ...)에는 cos을 사용한다.
    10000을 밑으로 하는 지수 함수로 주파수를 조절하여,
    차원마다 서로 다른 주기의 파형을 가지게 된다.
    """

    def __init__(self, d_model, max_len, device):
        """
            d_model : 모델의 임베딩 차원 수. 입력 토큰 임베딩과 같은 크기여야 덧셈이 가능함.
            max_len : 처리할 수 있는 시퀀스의 최대 길이. 이 길이만큼 인코딩을 미리 계산해 둠.
            device  : 텐서를 생성할 디바이스 ('cuda' or 'cpu').
        """
        super(PositionalEncoding, self).__init__()

        # (max_len, d_model) 크기의 영행렬로 인코딩 테이블 초기화
        # 입력 임베딩 행렬과 크기가 동일해야 element-wise 덧셈이 가능함
        self.encoding = torch.zeros(max_len, d_model, device=device)

        # Positional Encoding은 학습되는 값이 아니므로 gradient 계산 비활성화
        self.encoding.requires_grad = False

        # pos: 각 토큰의 위치 인덱스 생성 (0 ~ max_len-1)
        # shape: (max_len,) → unsqueeze로 (max_len, 1)로 변환
        # 2D로 만드는 이유: _2i (1D)와 브로드캐스팅 연산을 하기 위함
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)  # shape: (max_len, 1)

        # _2i: 수식의 2i에 해당하는 값. 짝수 인덱스만 추출 (step=2)
        # shape: (d_model/2,)
        # 예) d_model=512이면 _2i = [0, 2, 4, ..., 510]
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 짝수 차원(0, 2, 4, ...)에 sin 함수 적용
        # pos shape: (max_len, 1), _2i shape: (d_model/2,)
        # 브로드캐스팅 → 결과 shape: (max_len, d_model/2)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))

        # 홀수 차원(1, 3, 5, ...)에 cos 함수 적용
        # sin과 동일한 분모를 사용하되 cos을 적용
        # 결과 shape: (max_len, d_model/2)
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        """
        입력 시퀀스의 실제 길이(seq_len)에 맞게 Positional Encoding을 잘라서 반환.
        반환값은 토큰 임베딩(tok_emb)과 더해져 위치 정보가 주입된 최종 임베딩이 된다.

        변수:
            x : 토큰 인덱스 시퀀스. shape = (batch_size, seq_len)

        Returns:
            self.encoding[:seq_len, :] : shape = (seq_len, d_model)
                배치의 모든 샘플에 동일한 위치 인코딩이 적용되므로 batch 차원은 없음.
                이후 토큰 임베딩 (batch_size, seq_len, d_model)과 브로드캐스팅으로 덧셈됨.
        """
        # 입력에서 batch_size와 실제 시퀀스 길이(seq_len)를 추출
        # seq_len은 max_len보다 작거나 같으며, 실제 입력 길이만큼만 인코딩을 사용함
        batch_size, seq_len = x.size()

        # 미리 계산된 인코딩 테이블에서 seq_len 길이만큼 슬라이싱하여 반환
        # shape: (max_len, d_model) → (seq_len, d_model)
        # TransformerEmbedding에서 tok_emb (batch_size, seq_len, d_model)와 더해질 때
        # 브로드캐스팅에 의해 배치 전체에 동일하게 적용됨
        return self.encoding[:seq_len, :]