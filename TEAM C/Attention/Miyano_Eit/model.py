import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) #단어 index를 dense vector 로 변환
        self.birnn = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        #각 단어의 앞뒤 문맥을 모두 담은 annotation hj​를 생성


    def forward(self, x):
        # x: (batch, src_len)
        embedded = self.embedding(x)                          # (batch, src_len, embed_dim)
        outputs, hidden = self.birnn(embedded)                # outputs: (batch, src_len, hidden_dim*2)
        #outputs: 모든 시점의 hidden state. 
        #hidden: 양방향 마지막 시점의 hidden state (num_layers*2, batch, hidden_dim)
        
        # 양방향 마지막 hidden state를 합쳐서 decoder 초기 상태로 사용
        hidden = torch.tanh(
            torch.cat([hidden[-2], hidden[-1]], dim=1)
        ).unsqueeze(0)                                        # (1, batch, hidden_dim*2)

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.Wa = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        #논문의 Wa​ 행렬. Decoder의 이전 hidden state si−1​을 선형 변환
        self.Ua = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        #논문의 Ua​ 행렬. Encoder의 annotation hj​을 선형 변환
        self.va = nn.Linear(hidden_dim * 2, 1, bias=False)
        #논문의 va​ 벡터. Wa​와 Ua​의 결과를 tanh​로 활성화한 후 선형 변환하여 스칼라 에너지 eij​ 계산

    def forward(self, s_prev, encoder_outputs):
        # s_prev: 디코더 이전 hidden state.(batch, hidden_dim*2)
        # encoder_outputs: 인코더의 모든 annotation (batch, src_len, hidden_dim*2)

        src_len = encoder_outputs.size(1)

        s_prev = s_prev.unsqueeze(1).repeat(1, src_len, 1)   # (batch, src_len, hidden_dim*2)

        # e_ij = v^T tanh(Wa * s_prev + Ua * h_j)
        energy = self.va(torch.tanh(
            self.Wa(s_prev) + self.Ua(encoder_outputs)
        )).squeeze(2)                                         # (batch, src_len)

        alpha = F.softmax(energy, dim=1)                     # (batch, src_len)

        # c_i = sum alpha_ij * h_j
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden_dim*2)

        return context, alpha


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim * 2, hidden_dim * 2, batch_first=True)
        #RNN 입력은 현재 단어 임베딩과 attention으로 구한 context 벡터를 이어붙인 것
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        #RNN의 출력 hidden state를 단어 예측을 위한 선형 변환. 논문에서는 Wc​ 행렬에 해당

    def forward(self, y_prev, s_prev, encoder_outputs):
        # y_prev: (batch,)
        # s_prev: (1, batch, hidden_dim*2)

        embedded = self.embedding(y_prev).unsqueeze(1)       # (batch, 1, embed_dim)

        context, alpha = self.attention(
            s_prev.squeeze(0), encoder_outputs
        )                                                     # context: (batch, hidden_dim*2)

        rnn_input = torch.cat(
            [embedded, context.unsqueeze(1)], dim=2
        )                                                     # (batch, 1, embed_dim + hidden_dim*2)

        output, s_next = self.rnn(rnn_input, s_prev)         # output: (batch, 1, hidden_dim*2)

        pred = self.fc_out(output.squeeze(1))                 # (batch, vocab_size)

        return pred, s_next, alpha


class Seq2SeqAttention(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: (batch, src_len)
        # tgt: (batch, tgt_len)

        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)

        encoder_outputs, hidden = self.encoder(src)

        y_prev = tgt[:, 0]                                    # <SOS> token

        for t in range(1, tgt_len):
            pred, hidden, alpha = self.decoder(y_prev, hidden, encoder_outputs)
            outputs[:, t, :] = pred

            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            y_prev = tgt[:, t] if use_teacher else pred.argmax(1)

        return outputs