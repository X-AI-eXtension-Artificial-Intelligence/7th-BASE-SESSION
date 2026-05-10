import torch
import torch.nn as nn


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_pixels = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        preds = model(images)                           # (batch, num_classes, H, W)
        loss = criterion(preds, masks)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Pixel Accuracy 계산
        pred_labels = preds.argmax(dim=1)               # (batch, H, W)
        total_correct += (pred_labels == masks).sum().item()
        total_pixels += masks.numel()

    avg_loss = total_loss / len(dataloader)
    pixel_acc = total_correct / total_pixels * 100

    return avg_loss, pixel_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            total_loss += loss.item()
            pred_labels = preds.argmax(dim=1)
            total_correct += (pred_labels == masks).sum().item()
            total_pixels += masks.numel()

    return total_loss / len(dataloader), total_correct / total_pixels * 100