import torch
import torch.nn as nn
from model.attention import BahdanauAttention

class Decoder(nn.Module):
    """
    Bahdanau et al. Section 3.1, Eq.(4)

    s_i = f(s_{i-1}, y_{i-1}, c_i)
    p(y_i | ...) = g(y_{i-1}, s_i, c_i)
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers=1):
        super().__init__()
        self.attention = BahdanauAttention(hidden_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 입력: [embed; context] → GRU
        self.rnn = nn.GRU(
            embed_dim + hidden_size * 2,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt_token, decoder_hidden, encoder_outputs):
        """
        tgt_token      : (batch,)          이전 타임스텝 출력
        decoder_hidden : (batch, hidden)
        encoder_outputs: (batch, src_len, 2*hidden)
        """
        embedded = self.embedding(tgt_token).unsqueeze(1)  # (batch, 1, embed_dim)

        # Attention
        context, alpha = self.attention(decoder_hidden, encoder_outputs)
        context = context.unsqueeze(1)                     # (batch, 1, 2*hidden)

        # GRU 입력: [embed ; context]
        rnn_input = torch.cat([embedded, context], dim=2)  # (batch, 1, embed+2*hidden)
        output, hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        prediction = self.fc_out(output.squeeze(1))        # (batch, vocab_size)
        hidden = hidden.squeeze(0)                         # (batch, hidden)

        return prediction, hidden, alpha