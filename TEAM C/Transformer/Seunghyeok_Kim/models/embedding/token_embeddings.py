from torch import nn

# 토큰 임베딩 클래스 정의
class TokenEmbedding(nn.Embedding):                                                 # nn.Embedding 클래스를 상속받아 TokenEmbedding 클래스 정의

    def __init__(self, vocab_size, d_model):                                        # vocab_size와 d_model을 입력으로 받는 초기화 메서드 정의
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)    # nn.Embedding 클래스의 초기화 메서드를 호출하여 vocab_size, d_model, padding_idx를 설정
                                                    
