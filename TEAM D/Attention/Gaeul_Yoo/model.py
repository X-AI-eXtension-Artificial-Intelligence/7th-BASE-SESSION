import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        # batch_first=True면 입력이 [Batch, Seq, Feature] 순서여야 해
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        # input shape: [Batch, Seq]
        embedded = self.dropout(self.embedding(input)) # [Batch, Seq, Hidden]
        output, hidden = self.gru(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, query, values, mask):
        # query: [Batch, 1, Hidden], values: [Batch, Seq, Hidden]
        score = self.V(torch.tanh(self.W1(query) + self.W2(values))) # [Batch, Seq, 1]
        score = score.permute(0, 2, 1) # [Batch, 1, Seq]

        score.masked_fill_(mask.unsqueeze(1) == 0, -1e10)
        
        alphas = F.softmax(score, dim=-1)
        context = torch.bmm(alphas, values) # [Batch, 1, Hidden]
        return context, alphas

class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, input_mask, target_tensor=None, max_len=10):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(0) # SOS_token=0
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(max_len):
            decoder_output, decoder_hidden, _ = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, input_mask)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return F.log_softmax(decoder_outputs, dim=-1), decoder_hidden

    def forward_step(self, input, hidden, encoder_outputs, input_mask):
        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2) # [1, Batch, Hidden] -> [Batch, 1, Hidden]
        
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)
        
        # [Batch, 1, 2*Hidden]
        gru_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(gru_input, hidden)
        output = self.out(output)
        return output, hidden, attn_weights

class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderRNN(input_vocab_size, hidden_size)
        self.decoder = AttnDecoder(hidden_size, output_vocab_size)

    def forward(self, inputs, input_mask, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        return self.decoder(encoder_outputs, encoder_hidden, input_mask, targets)