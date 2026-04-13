import os
import numpy as np

import torch


class Dataset(torch.utils.data.Dataset):
    """
    U-Net 학습을 위한 Dataset 클래스

    역할:
    - npy 형태로 저장된 input / label 데이터 로드
    - segmentation task에 맞게 전처리 수행
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        # input / label 파일 분리
        self.lst_label = sorted([f for f in lst_data if f.startswith('label')])
        self.lst_input = sorted([f for f in lst_data if f.startswith('input')])

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        # npy 파일 로드
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # [0,255] → [0,1] 정규화 (학습 안정성 향상)
        label = label / 255.0
        input = input / 255.0

        # grayscale → channel dimension 추가
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        # transform 적용 (augmentation + tensor 변환)
        if self.transform:
            data = self.transform(data)

        return data


# =========================
# Transform 정의
# =========================

class ToTensor(object):
    """
    numpy → torch tensor 변환
    (H, W, C) → (C, H, W)
    """
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        return {
            'label': torch.from_numpy(label),
            'input': torch.from_numpy(input)
        }


class Normalization(object):
    """
    입력 이미지 정규화
    → mean/std 기반 scaling
    """
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        return {'label': label, 'input': input}


class RandomFlip(object):
    """
    Data Augmentation
    - 좌우 / 상하 뒤집기
    - segmentation에서는 input과 label 동일하게 변환해야 함
    """
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        return {'label': label, 'input': input}