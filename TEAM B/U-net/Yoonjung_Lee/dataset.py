import os
import numpy as np

import torch
import torch.nn as nn

## 데이터 로더를 구현하기(데이터 불러오는 규칙)
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir #폴더 경로
        self.transform = transform #변형시킬 도구
        
        # 지정된 경로의 모든 파일 목록을 가져옴
        lst_data = os.listdir(self.data_dir)
        #라벨과 결과 구분
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input
    # 데이터 개수 확인
    def __len__(self):
        return len(self.lst_label)
    #데이터 실제로 꺼내기 ( 한 세트)
    def __getitem__(self, index):
        # 파일 경로를 합쳐서 실제 데이터를 불러옴
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        
        # 0~255 사이의 숫자를 0~1 사이로 줄여줌 (AI가 계산하기 좋게)
        label = label/255.0
        input = input/255.0
        
        #만약 이미지가 평면(2D)이라면, AI가 인식할 수 있게 '채널'이라는 차원을 하나 추가함
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        # 이미지와 정답을 하나의 딕셔너리로 만듦
        data = {'input': input, 'label': label}
        
        #만약 변형 도구(transform)가 설정되어 있다면 적용함
        if self.transform:
            data = self.transform(data)

        return data


## 트렌스폼(ai에 넣기 위해 형태나 수치 변형) 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        
        # 이미지의 모양을 (가로, 세로, 채널)에서 (채널, 가로, 세로) -> (이게 파이토치 순서이기 때문에) 순서로 바꿈 (파이토치 규칙), 딥러닝은 보통 32비트 사용해서 이것
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)
        
        #넘파이(Numpy) 배열을 파이토치 텐서(Tensor)라는 전용 데이터 형식으로 변환
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data
        
# 수치 표준화(정규화)
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean #평균값
        self.std = std # 표준편차

    def __call__(self, data):
        label, input = data['label'], data['input']
        
        # 공식: (데이터 - 평균) / 표준편차 를 계산해서 숫자를 일정 범위로 맞춤
        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data
        
# 데이터를 많이 만들기 위한 뒤집기 함수
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        # [좌우 뒤집기 결정]
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)
        # [상하 뒤집기 결정]
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

