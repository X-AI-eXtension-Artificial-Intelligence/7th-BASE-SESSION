"""
models/layers/multi_head_attention.py
- Multi-Head Attention을 구현합니다.
"""

from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()

        # head 개수입니다.
        self.n_head = n_head

        # 전체 d_model 차원을 head별로 나눈 차원입니다.
        self.d_tensor = d_model // n_head

        # 실제 attention 계산 모듈입니다.
        self.attention = ScaleDotProductAttention()

        # 입력 x에서 Q, K, V를 각각 만들기 위한 선형 변환입니다.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 여러 head의 결과를 concat한 뒤 다시 d_model 차원으로 섞습니다.
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. q, k, v를 linear projection합니다.
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. 여러 head로 나누기 위해 shape를 바꿉니다.
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. head별 scaled dot-product attention을 계산합니다.
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. head들을 다시 하나로 합칩니다.
        out = self.concat(out)

        # 5. 마지막 linear layer로 head 간 정보를 섞습니다.
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """d_model 차원을 n_head개의 작은 차원으로 나눕니다."""
        # 입력 shape: [batch, seq_len, d_model]
        batch_size, length, d_model = tensor.size()

        # [batch, seq_len, head, d_tensor]
        tensor = tensor.view(batch_size, length, self.n_head, self.d_tensor)

        # attention 계산에 편하도록 [batch, head, seq_len, d_tensor]로 바꿉니다.
        tensor = tensor.transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """나뉘었던 여러 head 결과를 다시 d_model 차원으로 합칩니다."""
        # 입력 shape: [batch, head, seq_len, d_tensor]
        batch_size, head, length, d_tensor = tensor.size()

        # [batch, seq_len, head, d_tensor]
        tensor = tensor.transpose(1, 2).contiguous()

        # [batch, seq_len, d_model]
        tensor = tensor.view(batch_size, length, head * d_tensor)

        return tensor
