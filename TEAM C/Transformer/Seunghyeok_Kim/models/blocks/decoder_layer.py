from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

# Decoder Layer 클래스 구현
class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)    # MultiHeadAttention 클래스의 인스턴스를 생성하여 self.self_attention에 할당하여 셀프 어텐션에 사용
        self.norm1 = LayerNorm(d_model=d_model)     # LayerNorm 클래스의 인스턴스를 생성하여 self.norm1에 할당하여 첫 번째 레이어 정규화에 사용
        self.dropout1 = nn.Dropout(p=drop_prob)     # nn.Dropout 클래스의 인스턴스를 생성하여 self.dropout1에 할당하여 첫 번째 드롭아웃 레이어에 사용

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)     # LayerNorm 클래스의 인스턴스를 생성하여 self.norm2에 할당하여 두 번째 레이어 정규화에 사용
        self.dropout2 = nn.Dropout(p=drop_prob)     # nn.Dropout 클래스의 인스턴스를 생성하여 self.dropout2에 할당하여 두 번째 드롭아웃 레이어에 사용

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)     # PositionwiseFeedForward 클래스의 인스턴스를 생성하여 self.ffn에 할당하여 위치별 피드포워드 네트워크에 사용
        self.norm3 = LayerNorm(d_model=d_model)     # LayerNorm 클래스의 인스턴스를 생성하여 self.norm3에 할당하여 세 번째 레이어 정규화에 사용
        self.dropout3 = nn.Dropout(p=drop_prob)     # nn.Dropout 클래스의 인스턴스를 생성하여 self.dropout3에 할당하여 세 번째 드롭아웃 레이어에 사용

    def forward(self, dec, enc, trg_mask, src_mask):    # dec는 디코더의 입력 텐서 / enc는 인코더의 출력 텐서 / trg_mask는 디코더의 입력에 대한 마스크 / src_mask는 인코더의 출력에 대한 마스크
        # 1. compute self attention
        _x = dec                                                    
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)     # 셀프 어텐션 계산 : 디코더의 입력 텐서(dec)를 쿼리, 키, 값으로 사용하여 trg_mask를 마스크로 적용하여 어텐션 계산
        
        # 2. Residual Connection과 Layer Normalization
        x = self.dropout1(x)            # 드롭아웃 적용 : 어텐션 계산 결과에 드롭아웃을 적용하여 과적합 방지
        x = self.norm1(x + _x)          # 레이어 정규화 : 드롭아웃이 적용된 어텐션 계산 결과(x)와 원래의 입력 텐서(_x)를 더한 후, self.norm1을 이용하여 레이어 정규화를 수행

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)        # 인코더-디코더 어텐션 계산 : 디코더의 어텐션 계산 결과(x)를 쿼리로 사용하고, 인코더의 출력(enc)을 키와 값으로 사용하여 src_mask를 마스크로 적용하여 어텐션 계산
            
            # 4. Residual Connection과 Layer Normalization
            x = self.dropout2(x)        # 드롭아웃 적용 : 인코더-디코더 어텐션 계산 결과에 드롭아웃을 적용하여 과적합 방지
            x = self.norm2(x + _x)      # 레이어 정규화 : 드롭아웃이 적용된 인코더-디코더 어텐션 계산 결과(x)와 이전 레이어의 출력(_x)를 더한 후, self.norm2를 이용하여 레이어 정규화를 수행

        # 5. positionwise feed forward network
        _x = x                  # 위치별 피드포워드 네트워크 계산 : 현재 레이어의 출력(x)을 위치별 피드포워드 네트워크(self.ffn)에 입력하여 계산
        x = self.ffn(x)         # 위치별 피드포워드 네트워크 계산 결과를 x에 저장
        
        # 6. Residual Connection과 Layer Normalization
        x = self.dropout3(x)        # 드롭아웃 적용 : 위치별 피드포워드 네트워크 계산 결과에 드롭아웃을 적용하여 과적합 방지
        x = self.norm3(x + _x)      # 레이어 정규화 : 드롭아웃이 적용된 위치별 피드포워드 네트워크 계산 결과(x)와 이전 레이어의 출력(_x)를 더한 후, self.norm3를 이용하여 레이어 정규화를 수행
        return x
