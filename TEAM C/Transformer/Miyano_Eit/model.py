import torch
import torch.nn as nn
from model_parts.encoder import Encoder
from model_parts.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, h=8, d_ff=2048, N=6, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, h, d_ff, N, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, h, d_ff, N, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        tgt_len = tgt.size(1)
        pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        sub_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device)
        ).bool()
        return pad_mask & sub_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        encoder_output = self.encoder(src, src_mask)
        decoder_output, attn_weights = self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )
        output = self.fc_out(decoder_output)

        return output, attn_weights