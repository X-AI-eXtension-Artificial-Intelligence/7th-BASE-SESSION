import os
import numpy as np

import torch
import torch.nn as nn

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    """PyTorch의 DataLoader가 학습 루프에서 데이터를 자동으로 꺼내올 수 있도록 규격 맞춰주는 역할"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):

        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label/255.0 #npy로 저장된 이미지는 uint8 형식이라 픽셀값이 0~255 범위
        input = input/255.0

        if label.ndim == 2: #나중에 ToTensor에서 (H, W, C) → (C, H, W)로 바꿔야 하는데
            label = label[:, :, np.newaxis] #grayscale npy는 (H, W) shape이므로 채널 차원 추가
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


## 트렌스폼 구현하기 
"""PyTorch 포맷으로 변환"""
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32) # (H,W,C) → (C,H,W)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object): #정규화하는 이유? -> 입력값을 줄임으로써 gradient가 안정적으로 흘러 학습잘되게 하려고.
    def __init__(self, mean=0.5, std=0.5): #
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input'] #label은 정규화 안 함 → BCE loss 계산 시 label은 [0,1] 그대로 사용하므로 올바른 처리

        input = (input - self.mean) / self.std # mean=0.5, std=0.5 → [-1, 1]

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object): #flip하는 이유 -> 데이터부족 문제.
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5: #왜 label/input 동일하게 flip 하냐면 → 정답이 틀어지면 안 되기 때문
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

