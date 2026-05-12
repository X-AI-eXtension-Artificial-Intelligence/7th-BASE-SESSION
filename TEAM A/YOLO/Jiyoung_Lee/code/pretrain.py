import torch as th
import torch.nn as nn

import torchvision.transforms as transforms

import torch.optim as opt
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import DataLoader

from torchvision.datasets import ImageNet

import os

from model import YOLOv1

from tqdm import tqdm

from typing import Tuple, List


# =========================================
# Model Hyperparameters
# =========================================

S = 7
B = 2

# ImageNet class 개수
C = 1000


# =========================================
# Transform Hyperparameters
# =========================================

# resize 크기
RESIZE_D = 256

# 최종 input size
INPUT_D = 224


# =========================================
# DataLoader Hyperparameters
# =========================================

MINI_BATCH = 256

NUM_WORKERS = 5

SHUFFLE = True

PIN_MEMORY = True

DROP_LAST = True


# =========================================
# Training Hyperparameters
# =========================================

MAX_EPOCHS = 90

INIT_LR = 0.1

MOMENTUM = 0.9

WEIGHT_DECAY = 0.0001

PATIENCE = 2

MIN_LR = 0.0001


# =========================================
# ImageNet Dataset 경로
# =========================================

IMAGENET_DIR_PATH = "/home/soul/Development/datasets/ImageNet"


# =========================================
# Device 설정
# =========================================

DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'


# =========================================
# Checkpoint 설정
# =========================================

"""
TRAIN_MODEL:
    True -> 학습 수행
    False -> pretrained weight 평가만 수행
"""

TRAIN_MODEL = False

"""
LOAD_MODEL:
    True -> checkpoint 이어서 학습
    False -> 처음부터 학습
"""

LOAD_MODEL = True


CHECKPOINT_PATH = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object Detection/checkpoints/pretrain_checkpoint.pt"

PRETRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object Detection/checkpoints/pretrained_model_weights.pt"

CHECKPOINT_T = 1


# ============================================================
# Train One Epoch
# ============================================================

def train_epoch(
        train_loader: DataLoader,
        model: YOLOv1,
        optimizer: opt.SGD,
        criterion: nn.CrossEntropyLoss
) -> float:
    """
    한 epoch 학습
    """

    av_loss = 0.

    model.train()

    for x, y_gt in train_loader:

        # GPU 이동
        x = x.to(DEVICE)
        y_gt = y_gt.to(DEVICE)

        # prediction
        y_pred = model(x)

        # classification loss
        loss = criterion(y_pred, y_gt)

        # gradient 초기화
        optimizer.zero_grad()

        # backward
        loss.backward()

        # weight update
        optimizer.step()

        av_loss += loss.item()

    av_loss /= len(train_loader)

    return av_loss


# ============================================================
# Validation
# ============================================================

def validate_epoch(
        val_loader: DataLoader,
        model: YOLOv1,
        criterion: nn.CrossEntropyLoss
) -> float:
    """
    validation loss 계산
    """

    av_loss = 0.

    with th.no_grad():

        model.eval()

        for x, y_gt in val_loader:

            x = x.to(DEVICE)
            y_gt = y_gt.to(DEVICE)

            y_pred = model(x)

            loss = criterion(y_pred, y_gt)

            av_loss += loss.item()

    av_loss /= len(val_loader)

    return av_loss