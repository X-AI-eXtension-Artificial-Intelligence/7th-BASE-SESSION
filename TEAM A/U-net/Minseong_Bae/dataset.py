"""
Oxford-IIIT Pet Dataset 로더
- 37개 품종의 반려동물 이미지 + 픽셀 단위 segmentation mask
- mask 값: 1=foreground(pet), 2=background, 3=boundary → 이진 분류(pet vs background)로 변환
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


class PetSegDataset(Dataset):
    """Oxford Pet Dataset을 binary segmentation으로 변환"""
    def __init__(self, root, split='trainval', image_size=256, download=True):
        self.image_size = image_size

        # torchvision OxfordIIITPet - segmentation=True로 mask 포함
        self.base = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types='segmentation',
            download=download,
        )

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]

        img = self.img_transform(img)

        # mask: PIL Image, 값 1=pet, 2=background, 3=boundary
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        # binary mask: pet(1) vs 나머지(0)
        # 원본: 1=pet, 2=bg, 3=border → 1=pet, 0=bg(border포함)
        mask = (mask == 1).long()

        return img, mask


def get_dataloaders(root='./data', image_size=256, batch_size=8, val_ratio=0.2, num_workers=0,
                    max_train=None, max_test=None):
    """train/val/test DataLoader 반환. max_train/max_test로 샘플 수 제한 가능."""
    from torch.utils.data import Subset

    train_dataset = PetSegDataset(root, split='trainval', image_size=image_size)
    test_dataset = PetSegDataset(root, split='test', image_size=image_size)

    # 샘플 수 제한 (빠른 실험용)
    if max_train is not None and max_train < len(train_dataset):
        idx = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(42))[:max_train]
        train_dataset = Subset(train_dataset, idx.tolist())
    if max_test is not None and max_test < len(test_dataset):
        idx = torch.randperm(len(test_dataset), generator=torch.Generator().manual_seed(42))[:max_test]
        test_dataset = Subset(test_dataset, idx.tolist())

    # train/val split
    n_val = int(len(train_dataset) * val_ratio)
    n_train = len(train_dataset) - n_val
    train_set, val_set = random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders()
    imgs, masks = next(iter(train_loader))
    print(f"이미지 shape: {imgs.shape}")
    print(f"마스크 shape: {masks.shape}, 고유값: {masks.unique()}")
