# 기본 라이브러리 임포트
import os  # 파일 경로 및 디렉토리 처리용
import numpy as np  # 수치 계산 및 배열 처리용

import torch  # PyTorch 핵심 라이브러리
import torch.nn as nn  # 신경망 모듈 사용 (여기선 안 쓰이지만 일반적으로 함께 가져옴)

# -----------------------------
# Custom Dataset 클래스 정의
# -----------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir  # 데이터가 저장된 디렉토리 경로
        self.transform = transform  # transform 파이프라인 (optional)

        lst_data = os.listdir(self.data_dir)  # 디렉토리 내 모든 파일 리스트

        # 'label'로 시작하는 파일과 'input'으로 시작하는 파일을 각각 리스트로 분리
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        # 파일명 기준으로 정렬 (input과 label 매칭을 위해)
        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label  # 라벨 파일 리스트
        self.lst_input = lst_input  # 입력 이미지 파일 리스트

    def __len__(self):
        return len(self.lst_label)  # 데이터셋 전체 샘플 수 반환

    def __getitem__(self, index):
        # 현재 인덱스에 해당하는 label 및 input 파일 로드
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 픽셀 정규화 (0~255 → 0~1)
        label = label / 255.0
        input = input / 255.0

        # 채널 축 추가 (2D 이미지를 3D로 변경 → H x W x C)
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}  # 딕셔너리 형태로 구성

        # 트랜스폼 적용 (있는 경우)
        if self.transform:
            data = self.transform(data)

        return data  # 최종 데이터 반환


# -----------------------------
# Transform 정의
# -----------------------------

# NumPy 배열을 PyTorch 텐서로 변환하는 클래스
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # 채널 순서 변경: HWC → CHW (PyTorch 형식)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        # NumPy 배열을 Tensor로 변환
        data = {
            'label': torch.from_numpy(label),
            'input': torch.from_numpy(input)
        }

        return data


# 입력 이미지를 정규화하는 클래스
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean  # 평균값
        self.std = std  # 표준편차

    def __call__(self, data):
        label, input = data['label'], data['input']

        # input 이미지 정규화
        input = (input - self.mean) / self.std

        data = {
            'label': label,
            'input': input
        }

        return data


# 입력/라벨을 랜덤으로 좌우 또는 상하 반전하는 클래스 (Data Augmentation)
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # 좌우 반전
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        # 상하 반전
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {
            'label': label,
            'input': input
        }

        return data