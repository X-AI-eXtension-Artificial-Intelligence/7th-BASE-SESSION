import math
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention

    Q: Query  → 현재 단어 (기준)
    K: Key    → 비교 대상
    V: Value  → 실제 정보

    핵심 아이디어:
    Q와 K의 유사도를 계산 → softmax → 그 비율로 V를 가중합
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

        # 마지막 차원 기준 softmax (attention score → 확률)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        """
        q, k, v shape: [batch_size, head, length, d_k]

        mask:
        - padding mask or look-ahead mask
        """

        # k 기준으로 shape 가져옴
        batch_size, head, length, d_tensor = k.size()
        # d_tensor = d_k (key/query 차원)

        # ---------------------------------------------------
        # 1. QK^T 계산 (유사도 측정)
        # ---------------------------------------------------

        # k transpose → [batch, head, d_k, length]
        k_t = k.transpose(2, 3)

        # q @ k^T → [batch, head, length, length]
        # 각 단어가 다른 단어와 얼마나 관련있는지 (유사도)
        score = (q @ k_t) / math.sqrt(d_tensor)
        # scaling 이유:
        # d_k가 커지면 값이 너무 커져서 softmax가 saturate됨 → gradient 죽음 방지

        # ---------------------------------------------------
        # 2. Mask 적용
        # ---------------------------------------------------
        if mask is not None:
            # mask == 0인 부분은 매우 작은 값으로 만들어 softmax에서 0 되게 함
            score = score.masked_fill(mask == 0, -10000)

        # ---------------------------------------------------
        # 3. Softmax → 확률화
        # ---------------------------------------------------
        score = self.softmax(score)
        # 각 row 합 = 1
        # "각 단어가 다른 단어를 얼마나 볼지"

        # ---------------------------------------------------
        # 4. Value 가중합
        # ---------------------------------------------------
        # score @ v → weighted sum
        # 실제 정보(v)를 attention 비율(score)에 따라 합침
        out = score @ v
        # shape: [batch, head, length, d_k]

        # ---------------------------------------------------
        # 결과 반환
        # ---------------------------------------------------
        return out, score
        # out   → attention 결과 (다음 layer로 전달)
        # score → attention map (시각화 가능)