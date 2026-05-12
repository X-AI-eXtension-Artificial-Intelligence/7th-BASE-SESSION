"""
YOLOv1 학습/평가에 사용할 PASCAL VOC Detection 데이터셋을 불러오는 파일입니다.
이미지 파일과 CSV 형식의 bounding box annotation을 읽어서
모델이 사용할 수 있는 (image, target) 형태로 반환합니다.
"""

import torch as th
from torch.utils.data import Dataset
import os
import PIL.Image as Image
import csv
from typing import Callable, Optional, Tuple, Union, List


# PASCAL VOC detection 데이터셋을 PyTorch Dataset 형식으로 감싸는 클래스
class VOC_Detection(Dataset):
    """
    A custom Dataset for the VOC Detection data. An index number (starting from 0) and a color is assigned to each of
    the labels of the dataset.
    """
 # PASCAL VOC의 객체 class 개수
    C = 20

 # class index를 실제 class 이름으로 바꾸기 위한 리스트
    index2label = ["person",
                   "bird", "cat", "cow", "dog", "horse", "sheep",
                   "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
                   "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]

 # class 이름을 숫자 index로 바꾸기 위한 딕셔너리
    label2index = {label: index for index, label in enumerate(index2label)}

 # 예측 결과를 시각화할 때 class별 bounding box 색상
    label_clrs = ["#ff0000",
                  "#2e8b57", "#808000", "#800000", "#000080", "#2f4f4f", "#ffa500",
                  "#00ff00", "#ba55d3", "#00fa9a", "#00ffff", "#0000ff", "#f08080", "#ff00ff",
                  "#1e90ff", "#ffff54", "#dda0dd", "#ff1493", "#87cefa", "#ffe4c4"]

    def __init__(self, root_dir: str, split: str = 'train',
                 transforms: Optional[Callable] = None) -> None:
        """ Initialize the VOC_Detection Dataset object.

        :param root_dir: The root directory of the dataset (this directory contains two directories 'train/' and
                         'test/'.
        :param split: The split of the dataset ('train' or 'test')
        :param transforms: The transforms that are applied to the images (x) and their corresponding targets (y).
        """

 # split 값이 train/test 중 하나인지 확인
        assert split == 'train' or split == 'test'
        split_dir = os.path.join(root_dir, split)

 # 이미지와 정답 annotation이 저장된 경로를 설정
        self.img_dir = os.path.join(split_dir, "images")
        self.annot_dir = os.path.join(split_dir, "targets")
 # annotation 파일명에서 확장자를 제거하여 이미지 id 목록을 생성
        self.pseudonyms = [filename[:-4] for filename in os.listdir(self.annot_dir)]

        self.transforms = transforms

    def __len__(self) -> int:
        """
        Return the total number of instances of the dataset.

        :return: total instances of the dataset
        """
        return len(self.pseudonyms)

    def __getitem__(self, idx: int) -> Tuple[Union[th.Tensor, Image.Image], th.Tensor]:
        """
        Given an index number in range [0, dataset's length) , return the corresponding image and target of the dataset.
        If transforms is defined, the images and their targets are first transformed and then return by the function.

        :param idx: The given index number
        :return: The (x,y)-pair of the image and the target
        """
 # idx에 해당하는 이미지 id를 가져오기
        pid = self.pseudonyms[idx]
        img_path = os.path.join(self.img_dir, f'{pid}.jpg')
        annot_path = os.path.join(self.annot_dir, f'{pid}.csv')

 # 이미지를 PIL 형식으로 읽습니
        img = Image.open(img_path)
 # target은 [class_index, xmin, ymin, xmax, ymax] 형태로 저장
        target = []
        with open(annot_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)                    # CSV 첫 줄은 header이므로 건너뜁니다.
            for row in csv_reader:
                target.append([self.label2index[row[0]]] + [int(row[i]) for i in range(1, 5)])
 # list로 모은 annotation을 tensor로 변환
        target = th.Tensor(target)

 # transform이 주어진 경우 이미지와 bounding box를 함께 변환
        if self.transforms is not None:
            img, target = self.transforms((img, target))

        return img, target