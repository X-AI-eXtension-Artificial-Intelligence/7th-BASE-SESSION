import torch as th
from torch.utils.data import Dataset
import os
import PIL.Image as Image
import csv
from typing import Callable, Optional, Tuple, Union, List

class VOC_Detection(Dataset):
    """PASCAL VOC 객체 탐지 데이터셋을 다루는 커스텀 클래스"""
    
    C = 20 # 예측할 총 클래스(객체 종류) 개수

    # 20개의 객체 클래스 이름
    index2label = ["person", "bird", "cat", "cow", "dog", "horse", "sheep",
                   "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
                   "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]

    # 라벨(문자열) -> 인덱스(숫자) 변환 딕셔너리
    label2index = {label: index for index, label in enumerate(index2label)}

    # 바운딩 박스 시각화에 사용할 클래스별 고유 색상
    label_clrs = ["#ff0000", "#2e8b57", "#808000", "#800000", ...]

    def __init__(self, root_dir: str, split: str = 'train', transforms: Optional[Callable] = None) -> None:
        # 이미지 파일과 정답(Target) CSV 파일 경로 설정
        # ...
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[Union[th.Tensor, Image.Image], th.Tensor]:
        # 1. 인덱스에 해당하는 이미지와 정답 파일 읽기
        # 2. CSV 파일에서 정답 정보(클래스 인덱스, xmin, ymin, xmax, ymax) 추출
        # 3. transforms(데이터 증강 및 텐서 변환) 적용 후 반환
        # ...
        return img, target