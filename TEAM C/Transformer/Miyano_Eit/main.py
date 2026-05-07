import os
import torch
import torch.nn as nn
from model import Transformer
from data_loader import Vocabulary, get_dataloader
from train import train_epoch, evaluate

os.makedirs("etc", exist_ok=True)

# --- toy 데이터 ---
src_sentences = [
    "the cat sat on the mat",
    "a dog runs in the park",
    "she reads a book every day",
]
tgt_sentences = [
    "le chat s assis sur le tapis",
    "un chien court dans le parc",
    "elle lit un livre chaque jour",
]

# --- Vocabulary ---
src_vocab = Vocabulary()
tgt_vocab = Vocabulary()
src_vocab.build(src_sentences)
tgt_vocab.build(tgt_sentences)

# --- Dataloader ---
train_loader = get_dataloader(src_sentences, tgt_sentences,
                              src_vocab, tgt_vocab,
                              max_len=20, batch_size=2)

# --- 모델 설정 ---
D_MODEL = 128
H       = 4
D_FF    = 256
N       = 3
EPOCHS  = 200
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=D_MODEL,
    h=H,
    d_ff=D_FF,
    N=N,
    dropout=0.1,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

# --- 학습 ---
step = 0
for epoch in range(1, EPOCHS + 1):
    train_loss, step = train_epoch(model, train_loader, optimizer, criterion,
                                   device, step, D_MODEL)
    print(f"Epoch {epoch:02d} | Step {step:04d} | Train Loss: {train_loss:.4f}")

# --- 모델 저장 ---
torch.save(model.state_dict(), "etc/transformer.pt")
print("모델 저장 완료 → etc/transformer.pt")