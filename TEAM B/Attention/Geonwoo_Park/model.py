import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    """
    단방향 GRU encoder.
    논문은 BiRNN인데 여기는 forward만. -> 각 위치 representation이 미래 문맥 못 봄
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        # output: [B, Seq, D]   - 모든 timestep의 hidden (attention의 values로 쓰임)
        # hidden: [1, B, D]     - 마지막 timestep만 (decoder 초기 상태)
        return output, hidden


class DecoderRNN(nn.Module):
    """
    Attention 없는 baseline decoder.
    Bahdanau가 비판한 fixed-length bottleneck이 정확히 여기서 발생.
    encoder_hidden 한 개만 받아서 모든 target 단어 생성해야 함.
    """
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):
        batch_size = encoder_outputs.size(0)
        # 모든 batch에 SOS로 decoder 시작
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden  # TODO: bridge layer 고려 (linear projection)
        decoder_outputs = []

        for i in range(max_len):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            # 핵심 분기: target_tensor 있으면 teacher forcing, 없으면 greedy
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)  # [B, Seq, OutVocab]
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)                # embedding 위에 ReLU? 좀 특이함
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    """
    논문 alignment model 구현:
        e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)

    매핑:
        W1 = W_a   (decoder state 변환)
        W2 = U_a   (encoder annotation 변환)
        V  = v_a   (스칼라로 projection)
    """
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.W3 = nn.Linear(hidden_size, 1)    # dead code. 정의만 하고 안 씀

    def forward(self, query, values, mask):
        # query:  [B, 1, D]   현재 decoder state
        # values: [B, M, D]   encoder의 모든 출력 (M = source 길이)

        # Additive attention
        # W1(query): [B, 1, D],  W2(values): [B, M, D]  -> broadcasting으로 [B, M, D]
        # tanh -> V: [B, M, 1]
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values)))
        scores = scores.squeeze(2).unsqueeze(1)   # [B, M, 1] -> [B, 1, M]
                                                  # 이렇게 해놔야 나중에 bmm으로 context 만들 때 편함

        # 다른 score 함수들도 가능 (둘 다 query/values 차원 같아야 함):
        # Dot-Product:  scores = torch.bmm(query, values.permute(0,2,1))           # Luong 스타일
        # Cosine:       scores = F.cosine_similarity(query, values, dim=2).unsqueeze(1)

        # padding 위치를 -inf로 -> softmax 후 0 됨
        # 0으로 fill하면 안 됨 (exp(0)=1이라 여전히 비중 차지)
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))

        # 논문 식 (6): alpha_ij = softmax(e_ij)
        alphas = F.softmax(scores, dim=-1)

        # 논문 식 (5): c_i = sum_j alpha_ij * h_j
        # bmm: [B, 1, M] @ [B, M, D] = [B, 1, D]
        context = torch.bmm(alphas, values)

        return context, alphas


class AttnDecoder(nn.Module):
    """
    Attention 적용 decoder.
    핵심: GRU 입력 차원이 2*hidden_size (embedding과 context를 concat해서 넣음)

    NOTE: 논문 Eq.(4)에서는 c_i가 GRU의 reset/update/candidate gate 모두에 들어가는데
          여기서는 단순히 입력 단계에서 concat. 개념상 같지만 정확히 같지는 않음.
    """
    def __init__(self, hidden_size, output_size):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)   # 입력이 [embed; context]
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(max_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, input_mask)
            decoder_outputs.append(decoder_output)
            # attn_weights는 안 모음. 시각화하려면 여기 모아야 함

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)   # teacher forcing
            else:
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden


    def forward_step(self, input, hidden, encoder_outputs, input_mask):
        # query: GRU output 형식 [1, B, D] -> attention 입력 형식 [B, 1, D]
        query = hidden.permute(1, 0, 2)

        # 매 step마다 새 context 계산 (이게 Bahdanau의 핵심 - 동적 context vector c_i)
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)

        embedded = self.embedding(input)                  # [B, 1, D]
        attn = torch.cat((embedded, context), dim=2)      # [B, 1, 2D]
        output, hidden = self.gru(attn, hidden)
        output = self.out(output)
        return output, hidden, attn_weights


class EncoderDecoder(nn.Module):
    """
    Wrapper. decoder를 토글하면 attention 유/무 비교 ablation 가능.
    """
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderRNN(input_vocab_size, hidden_size)
        self.decoder = AttnDecoder(hidden_size, output_vocab_size)
        # self.decoder = DecoderRNN(hidden_size, output_vocab_size)   # baseline 비교용

    def forward(self, inputs, input_mask, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets)
        return decoder_outputs, decoder_hidden
