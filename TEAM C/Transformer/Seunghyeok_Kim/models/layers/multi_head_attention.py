from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention

# Multi-Head Attention 클래스 구현 
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):                # d_model : 모델의 차원, n_head : head의 수
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head                            # n_head는 멀티 헤드 어텐션에서 사용할 헤드의 수를 나타냅니다.
        self.attention = ScaleDotProductAttention()     # ScaleDotProductAttention 클래스의 인스턴스를 생성하여 self.attention에 할당합니다. 이 클래스는 스케일드 닷 프로덕트 어텐션을 구현하는 데 사용됩니다.
        self.w_q = nn.Linear(d_model, d_model)          # nn.Linear은 선형 변환을 수행하는 레이어입니다. 여기서는 입력 차원과 출력 차원이 모두 d_model인 선형 레이어를 생성하여 self.w_q에 할당합니다. 이 레이어는 쿼리 벡터를 변환하는 데 사용됩니다.
        self.w_k = nn.Linear(d_model, d_model)          # nn.Linear은 선형 변환을 수행하는 레이어입니다. 여기서는 입력 차원과 출력 차원이 모두 d_model인 선형 레이어를 생성하여 self.w_k에 할당합니다. 이 레이어는 키 벡터를 변환하는 데 사용됩니다.
        self.w_v = nn.Linear(d_model, d_model)          # nn.Linear은 선형 변환을 수행하는 레이어입니다. 여기서는 입력 차원과 출력 차원이 모두 d_model인 선형 레이어를 생성하여 self.w_v에 할당합니다. 이 레이어는 값 벡터를 변환하는 데 사용됩니다.
        self.w_concat = nn.Linear(d_model, d_model)     # nn.Linear은 선형 변환을 수행하는 레이어입니다. 여기서는 입력 차원과 출력 차원이 모두 d_model인 선형 레이어를 생성하여 self.w_concat에 할당합니다. 이 레이어는 멀티 헤드 어텐션의 출력을 변환하는 데 사용됩니다.

    def forward(self, q, k, v, mask=None):                          # q, k, v는 각각 쿼리, 키, 값 벡터를 나타내며, mask는 어텐션 계산에서 사용할 마스크입니다.
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)             # 1. linear layer를 통과시켜서 q, k, v를 구한다.

        q, k, v = self.split(q), self.split(k), self.split(v)       # 2. split 함수를 이용해서 q, k, v를 head의 수로 나눈다. (head의 수로 나눈다는 것은 head의 수만큼 병렬적으로 어텐션을 계산한다는 것을 의미한다.)

        out, attention = self.attention(q, k, v, mask=mask)         # 3. attention 함수를 이용해서 q, k, v를 이용해서 attention을 계산한다. (mask는 어텐션 계산에서 사용할 마스크이다.)

        out = self.concat(out)                                      # 4. concat 함수를 이용해서 out을 head의 수로 나눈 것을 다시 합친다. (head의 수로 나눈 것을 다시 합친다는 것은 병렬적으로 계산된 어텐션 결과를 하나로 합친다는 것을 의미한다.)
        out = self.w_concat(out)                                    # 5. linear layer를 통과시켜서 out을 구한다.

        return out

    def split(self, tensor):        # split 함수 : 입력 텐서를 헤드의 수로 나누는 함수

        batch_size, length, d_model = tensor.size()         # 입력 텐서의 크기를 batch_size, length, d_model로 언패킹

        d_tensor = d_model // self.n_head                   # d_tensor는 각 헤드에 할당되는 차원
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # 입력 텐서를 (batch_size, length, n_head, d_tensor) 형태로 변환한 후, transpose 함수를 이용해서 (batch_size, n_head, length, d_tensor) 형태로 변환

        return tensor

    def concat(self, tensor):       # concat 함수 : 헤드의 수로 나눈 텐서를 다시 합치는 함수

        batch_size, head, length, d_tensor = tensor.size()      # 입력 텐서의 크기를 batch_size, head, length, d_tensor로 언패킹
        d_model = head * d_tensor                               # d_model은 전체 모델의 차원으로, 각 헤드에 할당된 차원(d_tensor)과 헤드의 수(head)를 곱하여 계산

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)      # 입력 텐서를 (batch_size, length, n_head, d_tensor) 형태로 변환한 후, transpose 함수를 이용해서 (batch_size, length, n_head, d_tensor) 형태로 변환한 다음, view 함수를 이용해서 (batch_size, length, d_model) 형태로 변환
        return tensor
