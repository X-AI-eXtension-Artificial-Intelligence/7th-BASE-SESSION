
from torch import nn

from blocks.encoder_layer import EncoderLayer
from Transformer.embeddings.Transformer_Embedding import TransformerEmbedding


class Encoder(nn.Module):

    # enc_voc_size 단어장 갯수
    # max_len 최대 처리할수 있는 단어 수
    # d_model 단어 벡터의 차원
    # ffn_hidden 논문에선 2048
    # n_head 멀티 헤드의 갯수
    # n_layers N층수
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    #
    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x