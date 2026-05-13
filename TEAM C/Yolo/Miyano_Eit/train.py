import torch
from loss import YOLOLoss


def train_epoch(model, dataloader, optimizer, device, S=7, B=2, C=20):
    model.train()
    criterion = YOLOLoss(S=S, B=B, C=C)
    total_loss = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        preds = model(images)               # (batch, S, S, B*5+C)
        loss = criterion(preds, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, S=7, B=2, C=20):
    model.eval()
    criterion = YOLOLoss(S=S, B=B, C=C)
    total_loss = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss = criterion(preds, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)