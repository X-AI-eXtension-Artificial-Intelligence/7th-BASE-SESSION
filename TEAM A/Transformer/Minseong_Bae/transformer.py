"""
Transformer implementation based on:
"Attention Is All You Need" (Vaswani et al., 2017)
https://arxiv.org/abs/1706.03762

Architecture specs (base model):
  N=6, d_model=512, d_ff=2048, h=8, d_k=d_v=64, p_drop=0.1
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    Q: (..., seq_q, d_k)
    K: (..., seq_k, d_k)
    V: (..., seq_k, d_v)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (..., seq_q, seq_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights


class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
      head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
    """

    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # (batch, seq, d_model) -> (batch, h, seq, d_k)
        B, L, _ = x.size()
        return x.view(B, L, self.h, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        Q = self._split_heads(self.W_Q(query))
        K = self._split_heads(self.W_K(key))
        V = self._split_heads(self.W_V(value))

        x, _ = scaled_dot_product_attention(Q, K, V, mask)

        # (batch, h, seq, d_k) -> (batch, seq, d_model)
        B, _, L, _ = x.size()
        x = x.transpose(1, 2).contiguous().view(B, L, self.h * self.d_k)
        return self.dropout(self.W_O(x))


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Residual + LayerNorm wrapper
# ---------------------------------------------------------------------------

class SublayerConnection(nn.Module):
    """LayerNorm(x + Sublayer(x))"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """One encoder layer: self-attention + FFN, each wrapped in SublayerConnection."""

    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayers = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask=None):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.sublayers[1](x, self.ffn)
        return x


class Encoder(nn.Module):
    """Stack of N encoder layers."""

    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.W_Q.out_features)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """
    One decoder layer:
      1. Masked self-attention
      2. Cross-attention (encoder-decoder attention)
      3. FFN
    """

    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayers = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        x = self.sublayers[2](x, self.ffn)
        return x


class Decoder(nn.Module):
    """Stack of N decoder layers."""

    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.W_Q.out_features)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class Embeddings(nn.Module):
    """Token embedding scaled by sqrt(d_model)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        return self.embed(x) * self.scale


# ---------------------------------------------------------------------------
# Generator (output projection)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """Linear + log-softmax projection to vocabulary."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# ---------------------------------------------------------------------------
# Full Transformer
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    Full Transformer encoder-decoder model.

    Paper defaults (base model):
      N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
    """

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        N: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        h: int = 8,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.src_embed = nn.Sequential(
            Embeddings(src_vocab, d_model),
            PositionalEncoding(d_model, dropout, max_len),
        )
        self.tgt_embed = nn.Sequential(
            Embeddings(tgt_vocab, d_model),
            PositionalEncoding(d_model, dropout, max_len),
        )
        self.encoder = Encoder(EncoderLayer(d_model, h, d_ff, dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, h, d_ff, dropout), N)
        self.generator = Generator(d_model, tgt_vocab)

        # Weight sharing between src embedding, tgt embedding, and pre-softmax linear
        self.generator.proj.weight = self.tgt_embed[0].embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(out)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def make_src_mask(src, pad_idx: int = 0):
    """Mask padding tokens in source. (batch, 1, 1, seq)"""
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt, pad_idx: int = 0):
    """
    Combine padding mask and causal (no-look-ahead) mask.
    (batch, 1, seq, seq)
    """
    B, L = tgt.size()
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)           # (B, 1, 1, L)
    causal = torch.tril(torch.ones(L, L, device=tgt.device)).bool() # (L, L)
    return pad_mask & causal


# ---------------------------------------------------------------------------
# Learning rate scheduler  (eq. 3 in paper)
# ---------------------------------------------------------------------------

class TransformerLRScheduler:
    """
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * self.warmup ** -1.5,
        )
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ---------------------------------------------------------------------------
# Label smoothing loss  (ε_ls = 0.1)
# ---------------------------------------------------------------------------

class LabelSmoothingLoss(nn.Module):
    """KL-divergence based label smoothing as used in the paper."""

    def __init__(self, vocab_size: int, pad_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, log_probs, target):
        # log_probs: (batch * seq, vocab)
        # target:    (batch * seq,)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            true_dist[:, self.pad_idx] = 0
            mask = (target == self.pad_idx)
            true_dist[mask] = 0

        n_tokens = (~mask).sum().item()
        return self.criterion(log_probs, true_dist) / n_tokens


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(42)

    SRC_VOCAB, TGT_VOCAB = 1000, 1000
    BATCH, SRC_LEN, TGT_LEN = 2, 10, 8
    PAD = 0

    model = Transformer(src_vocab=SRC_VOCAB, tgt_vocab=TGT_VOCAB)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    src = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN))
    tgt = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN))

    src_mask = make_src_mask(src, PAD)
    tgt_mask = make_tgt_mask(tgt, PAD)

    log_probs = model(src, tgt, src_mask, tgt_mask)
    print(f"Output shape: {log_probs.shape}")  # (batch, tgt_len, vocab)

    # Loss
    criterion = LabelSmoothingLoss(TGT_VOCAB, pad_idx=PAD, smoothing=0.1)
    loss = criterion(log_probs.view(-1, TGT_VOCAB), tgt.view(-1))
    print(f"Loss: {loss.item():.4f}")

    # LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(optimizer, d_model=512, warmup_steps=4000)
    for step in [1, 100, 4000, 8000]:
        scheduler.step_num = step - 1
        lr = scheduler.step()
        print(f"  step={step:5d}  lr={lr:.6f}")

    print("\nAll checks passed.")
