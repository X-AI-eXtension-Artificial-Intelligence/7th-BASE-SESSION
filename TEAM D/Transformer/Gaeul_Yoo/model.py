import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class selfAttention(nn.Module):
    def __init__(self, embed_size, heads) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.queries_linear = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys_linear = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values_linear = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, value, key, query, mask):
        N_batch = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        value = value.reshape(N_batch, value_len, self.heads, self.head_dim)
        key = key.reshape(N_batch, key_len, self.heads, self.head_dim)
        query = query.reshape(N_batch, query_len, self.heads, self.head_dim)

        V = self.values_linear(value)
        K = self.keys_linear(key)
        Q = self.queries_linear(query)

        energy = torch.einsum("nqhd,nkhd->nhqk", [Q, K])

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, V]).reshape(
            N_batch, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
        super().__init__()
        self.attention = selfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # model.pth 파일에 저장된 'feed_forawrd' 오타를 그대로 유지합니다.
        self.feed_forawrd = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forawrd(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length, device) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)

        pos_embed = torch.zeros(max_length, embed_size)
        pos_embed.requires_grad = False
        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed = pos_embed.unsqueeze(0).to(device)

        self.layers = nn.ModuleList(
            [EncoderBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        _, seq_len = x.size()
        pos_embed = self.pos_embed[:, :seq_len, :]
        out = self.dropout(self.word_embedding(x) + pos_embed)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = selfAttention(embed_size, heads=heads)
        self.encoder_block = EncoderBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_trg_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.encoder_block(value, key, query, src_trg_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length, device) -> None:
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)

        pos_embed = torch.zeros(max_length, embed_size)
        pos_embed.requires_grad = False
        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed = pos_embed.unsqueeze(0).to(device)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_src, src_trg_mask, trg_mask):
        _, seq_len = x.size()
        pos_embed = self.pos_embed[:, :seq_len, :]
        out = self.dropout(self.word_embedding(x) + pos_embed).to(self.device)
        for layer in self.layers:
            out = layer(out, enc_src, enc_src, src_trg_mask, trg_mask)
        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size, num_layers, forward_expansion, heads, dropout, device, max_length) -> None:
        super().__init__()
        self.Encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length, device)
        self.Decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length, device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

    # [추가] test.py에서 사용하는 encode 메서드
    def encode(self, src):
        src_mask = self.make_pad_mask(src, src)
        return self.Encoder(src, src_mask)

    # [추가] test.py에서 사용하는 decode 메서드
    def decode(self, src, trg, enc_src):
        src_trg_mask = self.make_pad_mask(trg, src)
        trg_mask = self.make_trg_mask(trg)
        out = self.Decoder(trg, enc_src, src_trg_mask, trg_mask)
        out = self.fc_out(out)
        return F.log_softmax(out, dim=-1)

    def make_pad_mask(self, query, key):
        len_query, len_key = query.size(1), key.size(1)
        key_mask = key.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        key_mask = key_mask.repeat(1, 1, len_query, 1)
        query_mask = query.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        query_mask = query_mask.repeat(1, 1, 1, len_key)
        return key_mask & query_mask

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src)
        trg_mask = self.make_trg_mask(trg)
        src_trg_mask = self.make_pad_mask(trg, src)
        enc_src = self.Encoder(src, src_mask)
        out = self.Decoder(trg, enc_src, src_trg_mask, trg_mask)
        out = self.fc_out(out)
        return F.log_softmax(out, dim=-1)