from torch import nn
from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention

    핵심 아이디어:
    - 하나의 attention이 아니라 여러 개(head)로 나눠서 병렬 수행
    - 서로 다른 관점(subspace)에서 관계를 학습
    """

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head  # head 개수

        # 실제 attention 계산 (앞에서 본 Scaled Dot Product)
        self.attention = ScaleDotProductAttention()

        # Q, K, V를 각각 선형 변환
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # concat 후 다시 projection
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        입력 shape:
        q, k, v: [batch, length, d_model]
        """

        # ---------------------------------------------------
        # 1. Linear projection (Q, K, V 생성)
        # ---------------------------------------------------
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # shape: [batch, length, d_model]

        # ---------------------------------------------------
        # 2. head 개수만큼 split
        # ---------------------------------------------------
        q, k, v = self.split(q), self.split(k), self.split(v)
        # shape: [batch, head, length, d_k]

        # ---------------------------------------------------
        # 3. Attention 수행 (각 head별로)
        # ---------------------------------------------------
        out, attention = self.attention(q, k, v, mask=mask)
        # out shape: [batch, head, length, d_k]

        # ---------------------------------------------------
        # 4. concat (head 다시 합치기)
        # ---------------------------------------------------
        out = self.concat(out)
        # shape: [batch, length, d_model]

        # ---------------------------------------------------
        # 5. final linear projection
        # ---------------------------------------------------
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        tensor를 head 개수만큼 나누기

        input:
        [batch, length, d_model]

        output:
        [batch, head, length, d_k]
        """
        batch_size, length, d_model = tensor.size()

        # 각 head의 차원
        d_tensor = d_model // self.n_head

        # reshape + transpose
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor)
        tensor = tensor.transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        split된 tensor를 다시 합치기

        input:
        [batch, head, length, d_k]

        output:
        [batch, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()

        # 다시 원래 차원으로
        tensor = tensor.transpose(1, 2).contiguous()
        tensor = tensor.view(batch_size, length, head * d_tensor)

        return tensor