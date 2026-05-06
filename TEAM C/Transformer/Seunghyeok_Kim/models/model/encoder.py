from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

# Encoder 클래스 구현
class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)              # TransformerEmbedding 클래스의 인스턴스를 생성하여 self.emb에 할당하여 인코더 임베딩에 사용

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])     # EncoderLayer 클래스의 인스턴스를 n_layers 개수만큼 생성하여 self.layers에 할당하여 인코더 레이어에 사용

    def forward(self, x, src_mask):
        x = self.emb(x)                 # 인코더 임베딩 계산 : 인코더의 입력 텐서(x)를 self.emb에 입력하여 인코더 임베딩을 계산하여 x에 저장

        for layer in self.layers:       # 인코더 레이어 반복 : self.layers에 저장된 각 인코더 레이어(layer)에 대해 반복하여 인코더 레이어 계산을 수행
            x = layer(x, src_mask)      

        return x