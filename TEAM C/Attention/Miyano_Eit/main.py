import torch
import torch.nn as nn
from model import Seq2SeqAttention
from data_loader import Vocabulary, get_dataloader
from train import train_epoch, evaluate


# --- 간단한 toy 데이터 (실제 사용 시 dataset/ 폴더에서 로드) ---
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

# --- Vocabulary 구성 ---
src_vocab = Vocabulary()
tgt_vocab = Vocabulary()
src_vocab.build(src_sentences)
tgt_vocab.build(tgt_sentences)

# --- Dataloader ---
train_loader = get_dataloader(src_sentences, tgt_sentences,
                              src_vocab, tgt_vocab,
                              max_len=20, batch_size=2)

# --- 모델 설정 ---
EMBED_DIM  = 64
HIDDEN_DIM = 128
EPOCHS     = 20
LR         = 1e-3
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Seq2SeqAttention(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# --- 학습 ---
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}")

# --- 모델 저장 ---
torch.save(model.state_dict(), "etc/model.pt")
print("모델 저장 완료 → etc/model.pt")