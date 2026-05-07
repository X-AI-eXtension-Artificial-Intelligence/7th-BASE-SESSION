import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        dk = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        alpha = F.softmax(scores, dim=-1)
        output = torch.matmul(alpha, V)

        return output, alpha


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0

        self.h = h
        self.dk = d_model // h

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):
        batch, seq_len, d_model = x.size()
        x = x.view(batch, seq_len, self.h, self.dk)
        return x.transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        output, alpha = self.attention(Q, K, V, mask)

        batch, h, seq_len, dk = output.size()
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, h * dk)
        output = self.W_O(output)

        return output, alpha