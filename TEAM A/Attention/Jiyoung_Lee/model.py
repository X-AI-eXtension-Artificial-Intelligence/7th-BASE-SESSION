import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# Encoder
# 입력 문장을 hidden state로 압축하는 역할
# =========================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        # 단어 index → dense vector 변환
        self.embedding = nn.Embedding(input_dim, hidden_dim)

        # GRU: 순차 데이터 처리 (문맥 정보 학습)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x: [B, SeqLen]

        # embedding → [B, SeqLen, Hidden]
        embedded = self.embedding(x)

        # outputs: 모든 time step hidden
        # hidden: 마지막 hidden state
        outputs, hidden = self.gru(embedded)

        # outputs → Attention에서 사용
        # hidden → Decoder 초기 상태
        return outputs, hidden


# =========================================================
# Attention (Bahdanau)
# Decoder가 Encoder의 어떤 단어에 집중할지 결정
# =========================================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        # hidden + encoder_output → score 계산
        self.W = nn.Linear(hidden_dim * 2, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, B, H]
        # encoder_outputs: [B, SeqLen, H]

        batch_size, seq_len, _ = encoder_outputs.size()

        # hidden을 모든 time step에 맞게 반복
        # → [B, SeqLen, H]
        hidden = hidden.permute(1, 0, 2).repeat(1, seq_len, 1)

        # concat → [B, SeqLen, 2H]
        energy = torch.cat((hidden, encoder_outputs), dim=2)

        # score 계산
        energy = torch.tanh(self.W(energy))        # [B, SeqLen, H]
        scores = self.V(energy).squeeze(2)         # [B, SeqLen]

        # softmax → attention weight
        attn_weights = F.softmax(scores, dim=1)

        # context vector 계산
        # → weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        # [B, 1, H]

        return context, attn_weights


# =========================================================
# Decoder
# context + 이전 단어 → 다음 단어 예측
# =========================================================
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super().__init__()

        # 단어 embedding
        self.embedding = nn.Embedding(output_dim, hidden_dim)

        # context + embedding → GRU 입력
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)

        # 최종 단어 예측
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, context):
        # x: [B] (현재 입력 단어)
        # context: [B, 1, H]

        # embedding → [B, 1, H]
        embedded = self.embedding(x).unsqueeze(1)

        # context와 concat
        rnn_input = torch.cat((embedded, context), dim=2)

        # GRU
        output, hidden = self.gru(rnn_input, hidden)

        # FC → 단어 확률
        output = self.fc(output.squeeze(1))  # [B, vocab]

        return output, hidden


# =========================================================
# Seq2Seq (전체 구조)
# Encoder → Attention → Decoder
# =========================================================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, attention):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention

    def forward(self, src, trg):
        # src: [B, SrcLen]
        # trg: [B, TrgLen]

        # 1. Encoder
        encoder_outputs, hidden = self.encoder(src)

        outputs = []

        # 첫 입력은 <SOS>
        input = trg[:, 0]

        # 2. Decoder step-by-step
        for t in range(1, trg.size(1)):

            # Attention → context 생성
            context, attn_weights = self.attention(hidden, encoder_outputs)

            # Decoder → 다음 단어 예측
            output, hidden = self.decoder(input, hidden, context)

            outputs.append(output)

            # Teacher Forcing (정답을 다음 입력으로 사용)
            input = trg[:, t]

        # [B, SeqLen, vocab]
        return torch.stack(outputs, dim=1)