import torch
import torch.nn as nn
import torch.optim as optim
import random

from config import Config
from data.dataset import get_dataloader
from model.encoder import Encoder
from model.decoder import Decoder

def train():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    train_loader = get_dataloader(10000, cfg.seq_len, cfg.input_size, cfg.batch_size)
    val_loader   = get_dataloader(1000,  cfg.seq_len, cfg.input_size, cfg.batch_size, shuffle=False)

    encoder = Encoder(cfg.input_size, cfg.embed_dim, cfg.hidden_size).to(cfg.device)
    decoder = Decoder(cfg.output_size, cfg.embed_dim, cfg.hidden_size).to(cfg.device)

    params    = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.epochs + 1):
        encoder.train(); decoder.train()
        total_loss = 0

        for src, tgt in train_loader:
            src, tgt = src.to(cfg.device), tgt.to(cfg.device)
            batch_size = src.size(0)

            enc_outputs, dec_hidden = encoder(src)

            # Teacher forcing: 정답 토큰을 다음 스텝 입력으로 사용
            # 첫 입력은 SOS 역할로 0 사용
            dec_input = torch.zeros(batch_size, dtype=torch.long).to(cfg.device)

            loss = 0
            for t in range(cfg.seq_len):
                pred, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_outputs)
                loss    += criterion(pred, tgt[:, t])
                dec_input = tgt[:, t]   # teacher forcing

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() / cfg.seq_len

        avg_loss = total_loss / len(train_loader)

        # Validation
        val_acc = evaluate_accuracy(encoder, decoder, val_loader, cfg)
        print(f"Epoch {epoch:3d} | Loss {avg_loss:.4f} | Val Acc {val_acc:.2f}%")

    # 모델 저장
    torch.save(encoder.state_dict(), "encoder.pt")
    torch.save(decoder.state_dict(), "decoder.pt")
    print("모델 저장 완료")


def evaluate_accuracy(encoder, decoder, loader, cfg):
    encoder.eval(); decoder.eval()
    correct = total = 0

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(cfg.device), tgt.to(cfg.device)
            batch_size = src.size(0)

            enc_outputs, dec_hidden = encoder(src)
            dec_input = torch.zeros(batch_size, dtype=torch.long).to(cfg.device)

            preds = []
            for t in range(cfg.seq_len):
                pred, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_outputs)
                dec_input = pred.argmax(dim=1)
                preds.append(dec_input)

            preds = torch.stack(preds, dim=1)   # (batch, seq_len)
            correct += (preds == tgt).all(dim=1).sum().item()
            total   += batch_size

    return 100 * correct / total


if __name__ == "__main__":
    train()