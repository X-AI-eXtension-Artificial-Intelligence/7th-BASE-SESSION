"""
train.py
한 에폭 학습 및 평가 루프.

  train_one_epoch : 배치 순회, 역전파, 스케줄러 step
  evaluate        : @torch.no_grad() 평가
"""

import torch
import torch.nn as nn


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    cfg: dict,
) -> tuple[float, float]:
    """
    Returns:
        avg_loss (float), accuracy % (float)
    """
    device = cfg["device"]
    model.train()

    total_loss = 0.0
    correct    = 0
    total      = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()

        # 그래디언트 클리핑 — Transformer 학습 안정화
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    cfg: dict,
) -> tuple[float, float]:
    """
    Returns:
        avg_loss (float), accuracy % (float)
    """
    device = cfg["device"]
    model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy
