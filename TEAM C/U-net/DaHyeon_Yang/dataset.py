import os
import numpy as np

import torch
import torch.nn as nn

## 데이터 로더를 구현하기 → 데이터 파이프라인 구간에 들어섰음!
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # 디렉토리 내 모든 파일 목록을 가져와서 label과 input을 구분하여 정렬   
        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        # 데이터셋의 총 개수를 반환
        return len(self.lst_label)

    def __getitem__(self, index):
        # index에 해당하는 npy 파일을 로드
        # 메모리 효율적으로 학습 + 아끼기 가능함
        # 피룡할 때 조금씩 하나의 값만 추가해주는 과정
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        # 0~255인 픽셀 값을 0~1 사이로 정규화 (Min-Max Scaling)
        label = label/255.0
        input = input/255.0
        # 만약 (H, W) 형태라면 (H, W, 1)로 채널 차원 추가
        # 흑백 이미지의 경우 대체로 두 개의 값만 가지고 있는 경우가 많음, 그래서 차원을 맞춰주기 위해 1추가
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}
        # 아래에서 정의할 Transform(전처리) 적용
        if self.transform:
            data = self.transform(data)

        return data


## 트렌스폼 구현하기
# 데이터를 전처리 및 증강하는 과정(트랜스폼 = 데이터 증강)
# PyTorch 모델은 넘파이 배열을 처리하지 못해서 Tensor로 변환해줘야함
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        # 데이터 차원의 순서를 바꾸는 인덱스(0, 1, 2는 각각 세로, 가로, 채널을 의미)
        # 즉, 2번 자리에 있던 채널을 9번 맨 앞으로 보내라 등등
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data
# 데이터 값들이 너무 크거나 들쭉날쭉하면 안됨(즉, 수치 안정화)
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data
# 최종적으로 데이터 증강(좌우/상하로 뒤집으면서 이론상 400장의 효과 도출할 수 있음)
class RandomFlip(object):
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

