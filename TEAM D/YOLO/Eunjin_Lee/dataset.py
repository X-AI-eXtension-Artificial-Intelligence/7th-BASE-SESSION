"""
dataset.py - PASCAL VOC 객체탐지 데이터셋 클래스

PASCAL VOC 데이터셋을 PyTorch Dataset으로 래핑합니다.
- 20개 클래스 (사람, 동물, 탈것, 가구 등)
- 이미지와 CSV 형식의 어노테이션(바운딩 박스 + 클래스)을 로드
- 학습/테스트 분할 지원
"""

import torch as th
from torch.utils.data import Dataset
import os
import PIL.Image as Image
import csv
from typing import Callable, Optional, Tuple, Union, List


class VOC_Detection(Dataset):
    """
    PASCAL VOC 객체탐지 데이터셋.
    각 클래스에 인덱스 번호(0~19)와 시각화용 색상이 할당됩니다.
    """
    C = 20  # 총 클래스 수

    # 클래스 인덱스 → 클래스 이름 매핑
    index2label = ["person",
                   "bird", "cat", "cow", "dog", "horse", "sheep",
                   "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
                   "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]

    # 클래스 이름 → 인덱스 매핑 (역방향 조회용)
    label2index = {label: index for index, label in enumerate(index2label)}

    # 각 클래스별 시각화 색상 (바운딩 박스 표시용)
    label_clrs = ["#ff0000",
                  "#2e8b57", "#808000", "#800000", "#000080", "#2f4f4f", "#ffa500",
                  "#00ff00", "#ba55d3", "#00fa9a", "#00ffff", "#0000ff", "#f08080", "#ff00ff",
                  "#1e90ff", "#ffff54", "#dda0dd", "#ff1493", "#87cefa", "#ffe4c4"]

    def __init__(self, root_dir: str, split: str = 'train',
                 transforms: Optional[Callable] = None) -> None:
        """
        VOC_Detection 데이터셋 초기화.

        :param root_dir: 데이터셋 루트 디렉토리 (하위에 'train/', 'test/' 폴더 포함)
        :param split: 데이터 분할 ('train' 또는 'test')
        :param transforms: 이미지와 타겟에 적용할 변환 함수
        """
        assert split == 'train' or split == 'test'
        split_dir = os.path.join(root_dir, split)

        self.img_dir = os.path.join(split_dir, "images")       # 이미지 폴더 경로
        self.annot_dir = os.path.join(split_dir, "targets")    # 어노테이션 폴더 경로
        # 파일명에서 확장자를 제거하여 고유 ID 리스트 생성
        self.pseudonyms = [filename[:-4] for filename in os.listdir(self.annot_dir)]

        self.transforms = transforms

    def __len__(self) -> int:
        """
        데이터셋의 총 샘플 수를 반환합니다.

        :return: 데이터셋 크기
        """
        return len(self.pseudonyms)

    def __getitem__(self, idx: int) -> Tuple[Union[th.Tensor, Image.Image], th.Tensor]:
        """
        인덱스에 해당하는 이미지와 타겟(바운딩 박스)을 반환합니다.

        타겟 형식: (N, 5) 텐서 - 각 행: [클래스_인덱스, xmin, ymin, xmax, ymax]

        :param idx: 데이터 인덱스 (0 ~ len-1)
        :return: (이미지, 타겟) 튜플
        """
        pid = self.pseudonyms[idx]
        img_path = os.path.join(self.img_dir, f'{pid}.jpg')
        annot_path = os.path.join(self.annot_dir, f'{pid}.csv')

        # 이미지 로드
        img = Image.open(img_path)

        # CSV 어노테이션 파싱: [클래스명, xmin, ymin, xmax, ymax]
        target = []
        with open(annot_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # 헤더 행 건너뛰기
            for row in csv_reader:
                target.append([self.label2index[row[0]]] + [int(row[i]) for i in range(1, 5)])
        target = th.Tensor(target)

        # 변환 함수가 있으면 적용
        if self.transforms is not None:
            img, target = self.transforms((img, target))

        return img, target
