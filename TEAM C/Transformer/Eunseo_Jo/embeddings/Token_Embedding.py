import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    #vocab_size : 단어장의 수
    #d_model : 단어를 표현할 차원
    def __init__(self,vocab_size, d_model):
        #padding_idx 빈 토큰 학습 방지
        super(TokenEmbedding,self).__init__(vocab_size,d_model,padding_idx=1)