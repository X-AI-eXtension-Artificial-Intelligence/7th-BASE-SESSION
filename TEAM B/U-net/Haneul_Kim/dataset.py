## numpy 형태의 데이터를 불러와 정규화, tensor 변환, 데이터 증강 등을 적용하여
## PyTorch 학습에 적합한 형태로 제공하는 파일

import os
import numpy as np

import torch
import torch.nn as nn

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir # 데이터 폴더 경로
        self.transform = transform # 전처리/증강 함수

        lst_data = os.listdir(self.data_dir) # 폴더 안 목록 가져오기

        lst_label = [f for f in lst_data if f.startswith('label')] # 라벨 파일 분리
        lst_input = [f for f in lst_data if f.startswith('input')] # input 파일 분리

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self): # 데이터 개수 반환
        return len(self.lst_label) # DataLoader가 몇 개 있는지 알기 위해 필요

    def __getitem__(self, index): # 데이터 하나 가져오기
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label/255.0 # 정규화 (픽셀값으로 나눔)
        input = input/255.0

        if label.ndim == 2: # 채널 추가 (H, W) → (H, W, 1)
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform: # transform 적용 (전처리/augmentation 적용)
            data = self.transform(data)

        return data


## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32) # 채널 먼저 오도록 (H, W, C) → (C, H, W)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)} # numpy → tensor 변환

        return data

class Normalization(object): # 평균 0, 분산 1로 맞춤
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object): # 50% 확률로 뒤집기 → 과적합 방지
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

