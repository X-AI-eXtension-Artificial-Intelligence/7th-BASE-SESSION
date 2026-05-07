import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
                 embed_size=512, num_layers=6, forward_expansion=4, heads=8, dropout=0.1, max_length=100):
        super(TransformerModel, self).__init__()
        
        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.src_position_embedding = nn.Embedding(max_length, embed_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.trg_position_embedding = nn.Embedding(max_length, embed_size)
        
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=embed_size * forward_expansion,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        # (N, src_len) -> (N, 1, 1, src_len)
        src_mask = (src == self.src_pad_idx)
        return src_mask

    def forward(self, src, trg):
        src_seq_length, trg_seq_length = src.shape[1], trg.shape[1]
        
        src_positions = torch.arange(0, src_seq_length).unsqueeze(0).to(src.device)
        trg_positions = torch.arange(0, trg_seq_length).unsqueeze(0).to(src.device)
        
        src_emb = self.dropout(self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        trg_emb = self.dropout(self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        
        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(src.device)
        
        out = self.transformer(src_emb, trg_emb, tgt_mask=trg_mask, src_key_padding_mask=src_padding_mask)
        out = self.fc_out(out)
        return out