import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Encoder
# =========================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()

        # 단어를 dense vector로 변환 (one-hot → embedding)
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # GRU: 순차적으로 문장을 읽으면서 hidden state 생성
        self.rnn = nn.GRU(emb_dim, hidden_dim)

    def forward(self, src):
        # src: [seq_len, batch_size]

        # embedding 적용 → [seq_len, batch, emb_dim]
        embedded = self.embedding(src)

        # outputs: 모든 time step의 hidden state
        # hidden: 마지막 hidden state
        outputs, hidden = self.rnn(embedded)

        # outputs → Attention에서 사용됨
        # hidden → Decoder 초기 상태로 사용
        return outputs, hidden


# =========================
# Bahdanau Attention
# =========================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        # hidden + encoder_output을 결합해서 score 계산
        self.W = nn.Linear(hidden_dim * 2, hidden_dim)

        # attention score를 scalar로 변환
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch, hidden]
        # encoder_outputs: [seq_len, batch, hidden]

        seq_len = encoder_outputs.shape[0]

        # hidden을 seq_len만큼 복제해서 encoder_outputs와 크기 맞춤
        # → 모든 time step과 비교하기 위함
        hidden = hidden.repeat(seq_len, 1, 1)

        # concat: decoder hidden + encoder hidden
        # → "얼마나 중요한지" 계산하기 위한 입력
        energy = torch.tanh(
            self.W(torch.cat((hidden, encoder_outputs), dim=2))
        )

        # attention score 계산 → [seq_len, batch, 1]
        attention = self.v(energy).squeeze(2)

        # softmax 적용 → 중요도 확률 분포
        # dim=0: sequence 방향 기준으로 normalize
        return F.softmax(attention, dim=0)


# =========================
# Decoder
# =========================
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        # 이전 단어를 embedding으로 변환
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # 입력: (embedding + context vector)
        self.rnn = nn.GRU(emb_dim + hidden_dim, hidden_dim)

        # 최종 단어 예측 layer
        self.fc_out = nn.Linear(hidden_dim * 2 + emb_dim, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size]
        input = input.unsqueeze(0)  # → [1, batch]

        # embedding → [1, batch, emb_dim]
        embedded = self.embedding(input)

        # =========================
        # Attention 계산
        # =========================

        # attention weight → [seq_len, batch]
        attn = self.attention(hidden, encoder_outputs)

        # [seq_len, batch, 1] 형태로 변환
        attn = attn.unsqueeze(2)

        # context vector 계산
        # → encoder_outputs에 weight를 곱해서 중요한 정보만 추출
        # 결과: [1, batch, hidden]
        context = torch.sum(attn * encoder_outputs, dim=0, keepdim=True)

        # =========================
        # Decoder RNN
        # =========================

        # embedding + context 합쳐서 RNN 입력으로 사용
        rnn_input = torch.cat((embedded, context), dim=2)

        # RNN 수행
        output, hidden = self.rnn(rnn_input, hidden)

        # 차원 정리
        output = output.squeeze(0)
        context = context.squeeze(0)
        embedded = embedded.squeeze(0)

        # 최종 단어 예측
        output = self.fc_out(
            torch.cat((output, context, embedded), dim=1)
        )

        return output, hidden


# =========================
# Seq2Seq 전체 모델
# =========================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch]
        # trg: [trg_len, batch]

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # 결과 저장 tensor
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # =========================
        # Encoder 실행
        # =========================
        encoder_outputs, hidden = self.encoder(src)

        # 첫 입력은 <sos>
        input = trg[0, :]

        # =========================
        # Decoder 반복
        # =========================
        for t in range(1, trg_len):

            # 이전 단어 + hidden + encoder 정보 → 다음 단어 예측
            output, hidden = self.decoder(input, hidden, encoder_outputs)

            outputs[t] = output

            # teacher forcing 결정
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            # 가장 높은 확률 단어 선택
            top1 = output.argmax(1)

            # 다음 입력 결정
            # teacher forcing: 정답 사용
            # 아니면: 모델 예측 사용
            input = trg[t] if teacher_force else top1

        return outputs
