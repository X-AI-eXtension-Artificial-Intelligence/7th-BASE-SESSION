import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Bahdanau et al. Section 3.2
    BiRNN으로 각 입력 단어의 annotation h_j를 생성
    h_j = [h_forward_j ; h_backward_j]
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(
            embed_dim,
            hidden_size,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
        )
        # BiRNN의 출력(2*hidden)을 hidden으로 줄여 decoder 초기 상태에 사용
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)                  # (batch, seq_len, embed_dim)
        outputs, hidden = self.rnn(embedded)
        # outputs: (batch, seq_len, 2*hidden)  ← annotation sequence
        # hidden:  (2*n_layers, batch, hidden)

        # forward/backward 마지막 은닉 상태를 합쳐 decoder 초기 상태 생성
        hidden_fwd = hidden[-2]   # forward  last hidden
        hidden_bwd = hidden[-1]   # backward last hidden
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        decoder_hidden = torch.tanh(self.fc(hidden_cat))  # (batch, hidden)

        return outputs, decoder_hidden