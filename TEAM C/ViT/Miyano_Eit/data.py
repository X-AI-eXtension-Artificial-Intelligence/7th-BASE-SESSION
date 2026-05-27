"""
data.py
CIFAR-10 데이터셋 로드 및 전처리.

train : RandomCrop + HorizontalFlip + ColorJitter + Normalize
test  : Normalize only
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# CIFAR-10 채널별 평균/표준편차 (학습셋 기준)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transforms(img_size: int):
    """학습/평가 transform 반환."""
    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4
        ),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_tf, test_tf


def get_dataloaders(cfg: dict):
    """
    CIFAR-10 DataLoader 반환.

    Returns:
        train_loader, test_loader
    """
    train_tf, test_tf = get_transforms(cfg["img_size"])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,
        download=True, transform=train_tf,
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False,
        download=True, transform=test_tf,
    )

    train_loader = DataLoader(
        train_set,
        batch_size  = cfg["batch_size"],
        shuffle     = True,
        num_workers = cfg["num_workers"],
        pin_memory  = True,
        drop_last   = True,      # 배치 크기 일정 유지
    )
    test_loader = DataLoader(
        test_set,
        batch_size  = cfg["batch_size"],
        shuffle     = False,
        num_workers = cfg["num_workers"],
        pin_memory  = True,
    )

    print(f"  Train: {len(train_set):,} samples  "
          f"({len(train_loader)} batches)")
    print(f"  Test : {len(test_set):,} samples  "
          f"({len(test_loader)} batches)")

    return train_loader, test_loader
