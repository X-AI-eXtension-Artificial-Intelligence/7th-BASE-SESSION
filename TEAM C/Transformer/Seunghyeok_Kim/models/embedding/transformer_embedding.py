from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding

# Transformer 모델에서 단어 임베딩과 포지셔널 인코딩을 결합하여 단어의 위치 정보를 포함하는 임베딩을 생성하는 클래스 정의
class TransformerEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):    # vocab_size, d_model, max_len, drop_prob, device를 입력으로 받는 초기화 메서드 정의

        super(TransformerEmbedding, self).__init__()                        
        self.tok_emb = TokenEmbedding(vocab_size, d_model)                  # TokenEmbedding 클래스의 인스턴스를 생성하여 self.tok_emb에 저장. vocab_size와 d_model을 입력으로 전달하여 단어 임베딩을 초기화
        self.pos_emb = PositionalEncoding(d_model, max_len, device)         # PositionalEncoding 클래스의 인스턴스를 생성하여 self.pos_emb에 저장. d_model, max_len, device를 입력으로 전달하여 포지셔널 인코딩을 초기화
        self.drop_out = nn.Dropout(p=drop_prob)                             # nn.Dropout 클래스의 인스턴스를 생성하여 self.drop_out에 저장. drop_prob을 입력으로 전달하여 드롭아웃 확률을 설정

    def forward(self, x):                           # 입력 텐서 x를 받아서 단어 임베딩과 포지셔널 인코딩을 결합하여 최종 임베딩을 반환하는 forward 메서드 정의
        tok_emb = self.tok_emb(x)                   # 입력 텐서 x를 self.tok_emb에 전달하여 단어 임베딩을 계산하여 tok_emb에 저장. tok_emb의 크기는 (batch_size, seq_len, d_model)임
        pos_emb = self.pos_emb(x)                   # 입력 텐서 x를 self.pos_emb에 전달하여 포지셔널 인코딩을 계산하여 pos_emb에 저장. pos_emb의 크기는 (seq_len, d_model)임
        return self.drop_out(tok_emb + pos_emb)     # 단어 임베딩과 포지셔널 인코딩을 element-wise로 더한 후에 self.drop_out에 전달하여 드롭아웃을 적용한 최종 임베딩을 반환. 최종 임베딩의 크기는 (batch_size, seq_len, d_model)임
