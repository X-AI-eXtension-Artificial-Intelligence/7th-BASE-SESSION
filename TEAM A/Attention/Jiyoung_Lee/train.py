import torch
import torch.nn as nn
import torch.optim as optim

from model import Encoder, Decoder, Attention, Seq2Seq
from load_data import get_dataloader

# =========================
# 설정
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "./data-2/eng-fra.txt"

# =========================
# 데이터 로드
# =========================
input_lang, output_lang, loader = get_dataloader(DATA_PATH)

# =========================
# 모델 구성
# =========================
encoder = Encoder(input_lang.n_words, 256)
attention = Attention(256)
decoder = Decoder(output_lang.n_words, 256)

model = Seq2Seq(encoder, decoder, attention).to(device)

# =========================
# Loss / Optimizer
# =========================
criterion = nn.CrossEntropyLoss(ignore_index=0)  # padding 무시
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# 학습
# =========================
for epoch in range(10):
    total_loss = 0

    for src, trg in loader:
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output = output.reshape(-1, output.size(-1))
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)

        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =========================
# 샘플 출력
# =========================
print("\nSample check")

for src, trg in loader:
    print("SRC:", src[0])
    print("TRG:", trg[0])
    break