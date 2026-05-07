import torch.nn as nn
import Transformer.embeddings.Token_Embedding as Token_Embedding
import Transformer.embeddings.Position_Embedding as Position_Embedding

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb=Token_Embedding(vocab_size,d_model)
        self.pos_emb=Position_Embedding(d_model,max_len,device)
        self.drop_out=nn.Dropout(p=drop_prob)

    def forward(self):
        #단순히 더해주기 + 드롭아웃
        return self.drop_out(self.tok_emb+self.pos_emb)