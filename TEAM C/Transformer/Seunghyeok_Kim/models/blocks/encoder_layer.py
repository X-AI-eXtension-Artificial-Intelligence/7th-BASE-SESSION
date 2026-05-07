from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

# Encoder Layer 클래스 구현
class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)     # MultiHeadAttention 클래스의 인스턴스를 생성하여 self.attention에 할당
        self.norm1 = LayerNorm(d_model=d_model)     # LayerNorm 클래스의 인스턴스를 생성 self.norm1에 할당하여 첫 번째 레이어 정규화에 사용
        self.dropout1 = nn.Dropout(p=drop_prob)     # nn.Dropout 클래스의 인스턴스를 생성하여 self.dropout1에 할당하여 첫 번째 드롭아웃 레이어에 사용

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)     # PositionwiseFeedForward 클래스의 인스턴스를 생성하여 self.ffn에 할당하여 위치별 피드포워드 네트워크에 사용
        self.norm2 = LayerNorm(d_model=d_model)     # LayerNorm 클래스의 인스턴스를 생성하여 self.norm2에 할당하여 두 번째 레이어 정규화에 사용
        self.dropout2 = nn.Dropout(p=drop_prob)     # nn.Dropout 클래스의 인스턴스를 생성하여 self.dropout2에 할당하여 두 번째 드롭아웃 레이어에 사용

    def forward(self, x, src_mask):
        # 1. 셀프 어텐션 계산
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. Residual Connection과 Layer Normalization
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. 위치별 피드포워드 네트워크
        _x = x
        x = self.ffn(x)
      
        # 4. Residual Connection과 Layer Normalization
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
