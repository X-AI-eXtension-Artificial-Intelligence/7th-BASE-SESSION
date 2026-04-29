import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Bahdanau et al. Section 3.1, Eq.(5)(6)

    e_ij = a(s_{i-1}, h_j)          alignment model
         = v_a^T * tanh(W_a * s_{i-1} + U_a * h_j)

    alpha_ij = softmax(e_ij)         attention weight

    c_i = sum_j alpha_ij * h_j       context vector
    """
    def __init__(self, hidden_size):
        super().__init__()
        # decoder hidden: hidden_size
        # encoder output: hidden_size * 2 (BiRNN)
        self.W_a = nn.Linear(hidden_size,     hidden_size, bias=False)
        self.U_a = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, 1,  bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden  : (batch, hidden)
        encoder_outputs : (batch, src_len, 2*hidden)
        """
        # s_{i-1} 변환: (batch, hidden) → (batch, 1, hidden) → broadcast
        dec = self.W_a(decoder_hidden).unsqueeze(1)        # (batch, 1, hidden)
        enc = self.U_a(encoder_outputs)                    # (batch, src_len, hidden)

        # energy e_ij
        energy = self.v_a(torch.tanh(dec + enc))           # (batch, src_len, 1)
        energy = energy.squeeze(2)                         # (batch, src_len)

        # attention weight alpha_ij
        alpha = F.softmax(energy, dim=1)                   # (batch, src_len)

        # context vector c_i
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs)  # (batch, 1, 2*hidden)
        context = context.squeeze(1)                              # (batch, 2*hidden)

        return context, alpha