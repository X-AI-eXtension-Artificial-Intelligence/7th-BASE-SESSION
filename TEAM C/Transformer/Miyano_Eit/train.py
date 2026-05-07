import torch
import torch.nn as nn


def get_lr(step, d_model, warmup_steps=4000):
    if step == 0:
        step = 1
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def train_epoch(model, dataloader, optimizer, criterion, device, step, d_model):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output, _ = model(src, tgt_input)

        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        step += 1
        lr = get_lr(step, d_model)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader), step


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output, _ = model(src, tgt_input)

            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()

    return total_loss / len(dataloader)