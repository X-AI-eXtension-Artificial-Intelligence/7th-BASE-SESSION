import torch
import torch.nn as nn


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()

        output = model(src, tgt)
        # output: (batch, tgt_len, vocab_size)
        # tgt:    (batch, tgt_len)

        output = output[:, 1:, :].reshape(-1, output.size(-1))
        tgt = tgt[:, 1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, teacher_forcing_ratio=0.0)

            output = output[:, 1:, :].reshape(-1, output.size(-1))
            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt)
            total_loss += loss.item()

    return total_loss / len(dataloader)