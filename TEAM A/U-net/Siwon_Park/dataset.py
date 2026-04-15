"""
U-Net Dataset & Transforms
논문: "we use excessive data augmentation by applying elastic deformations"
    → 본 구현은 flip, elastic deformation, random crop 포함
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import map_coordinates, gaussian_filter


class SegmentationDataset(Dataset):
    """
    npy 형식의 input / label 쌍을 로드하는 Dataset

    논문 전처리:
    - 픽셀값 [0,255] → [0,1] 정규화
    - 채널 차원 추가 (H,W) → (1,H,W)
    """

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        files = os.listdir(data_dir)
        self.labels = sorted(f for f in files if f.startswith('label'))
        self.inputs = sorted(f for f in files if f.startswith('input'))

        assert len(self.labels) == len(self.inputs), \
            f"label({len(self.labels)})과 input({len(self.inputs)}) 수가 다릅니다."

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = np.load(os.path.join(self.data_dir, self.labels[idx])).astype(np.float32)
        image = np.load(os.path.join(self.data_dir, self.inputs[idx])).astype(np.float32)

        # [0,255] → [0,1]
        label = label / 255.0
        image = image / 255.0

        # (H,W) → (H,W,1) : transform이 (H,W,C) 가정
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

class Normalize:
    """
    논문 기준: 이미 [0,1]이므로 mean/std 기반 추가 정규화
    (ImageNet pretrain 없이 from-scratch이므로 0.5/0.5 사용)
    """
    def __init__(self, mean: float = 0.5, std: float = 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = (image - self.mean) / self.std
        # label은 0/1 binary mask이므로 정규화 하지 않음
        return {'image': image, 'label': label}


class RandomFlip:
    """
    논문 augmentation: horizontal / vertical flip
    image와 label 반드시 동일하게 변환
    """
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()

        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()
            label = np.flipud(label).copy()

        return {'image': image, 'label': label}


class ElasticDeformation:
    """
    논문 핵심 augmentation:
    "The key challenge in biomedical image segmentation is the very limited
     availability of annotated images ... we use excessive data augmentation
     by applying elastic deformations"

    Gaussian 필터로 부드러운 변위 필드(displacement field)를 생성 후
    image와 label에 동일하게 적용
    """
    def __init__(self, alpha: float = 34.0, sigma: float = 4.0, p: float = 0.5):
        self.alpha = alpha   # 변형 강도 (논문 권장: 34)
        self.sigma = sigma   # 변형 부드러움 (논문 권장: 4)
        self.p = p

    def __call__(self, sample):
        if np.random.rand() > self.p:
            return sample

        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]

        # 랜덤 변위 필드 생성 후 Gaussian으로 부드럽게
        dx = gaussian_filter(np.random.randn(h, w) * self.alpha, self.sigma)
        dy = gaussian_filter(np.random.randn(h, w) * self.alpha, self.sigma)

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        coords_x = np.clip(x + dx, 0, w - 1)
        coords_y = np.clip(y + dy, 0, h - 1)

        def _deform(arr):
            out = np.zeros_like(arr)
            for c in range(arr.shape[2]):
                out[:, :, c] = map_coordinates(
                    arr[:, :, c],
                    [coords_y.ravel(), coords_x.ravel()],
                    order=1, mode='reflect'
                ).reshape(h, w)
            return out

        return {'image': _deform(image), 'label': _deform(label)}


class ToTensor:
    """
    (H, W, C) numpy → (C, H, W) torch.Tensor
    """
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        label = torch.from_numpy(label.transpose(2, 0, 1).copy()).float()

        return {'image': image, 'label': label}
