import torch as th
from torch.utils.data import Dataset
import os
import PIL.Image as Image
import csv
from typing import Callable, Optional, Tuple, Union, List

# 관련 데이터와 동작을 하나로 묶은 클래스
class VOC_Detection(Dataset):
    """
    AVOC Detection 데이터를 위한 커스텀 Dataset 클래스.
    각 label에는 index 번호가 부여되어 있고,
    시각화 등에 사용할 color도 정의되어 있음. custom Dataset for the VOC Detection data. An index number (starting from 0) and a color is assigned to each of
    the labels of the dataset.
    """

    # 클래스 개수
    # VOC 데이터셋은 20개의 object 클래스를 가짐
    C = 20

    # index -> label 변환용 리스트
    index2label = ["person",
                   "bird", "cat", "cow", "dog", "horse", "sheep",
                   "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
                   "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]

    # label -> index 변환용 딕셔너리(모델이 label을 직접 학습할 수 없어서)
    label2index = {label: index for index, label in enumerate(index2label)}

    # bounding box를 시각화할 때 사용 가능
    label_clrs = ["#ff0000",
                  "#2e8b57", "#808000", "#800000", "#000080", "#2f4f4f", "#ffa500",
                  "#00ff00", "#ba55d3", "#00fa9a", "#00ffff", "#0000ff", "#f08080", "#ff00ff",
                  "#1e90ff", "#ffff54", "#dda0dd", "#ff1493", "#87cefa", "#ffe4c4"]

    def __init__(self, root_dir: str, split: str = 'train',
                 transforms: Optional[Callable] = None) -> None:
        """ Dataset 객체 초기화 함수.

        root_dir 아래에 train/test 폴더가 있고,
        각 split 폴더 안에는 images/와 targets/가 있다고 가정함.

        :param split: The split of the dataset ('train' or 'test')
        :param transforms: The transforms that are applied to the images (x) and their corresponding targets (y).
        """

        assert split == 'train' or split == 'test'
        split_dir = os.path.join(root_dir, split)

        self.img_dir = os.path.join(split_dir, "images")
        self.annot_dir = os.path.join(split_dir, "targets")
        self.pseudonyms = [filename[:-4] for filename in os.listdir(self.annot_dir)]

        self.transforms = transforms

    def __len__(self) -> int:
        """
        Dataset에 들어있는 전체 데이터 개수를 반환.

        : DataLoader는 이 값을 이용해서 전체 batch 수를 계산함.
        """
        return len(self.pseudonyms)

    def __getitem__(self, idx: int) -> Tuple[Union[th.Tensor, Image.Image], th.Tensor]:
        """
        idx에 해당하는 이미지와 정답 target을 반환하는 함수.

        PyTorch DataLoader는 내부적으로 이 함수를 반복 호출해서 mini-batch를 구성함.

        :param idx: The given index number
        return:
            img: PIL Image 또는 transform 후 Tensor
            target: Tensor 형태의 bounding box annotation
        """
        pid = self.pseudonyms[idx]
        img_path = os.path.join(self.img_dir, f'{pid}.jpg')
        annot_path = os.path.join(self.annot_dir, f'{pid}.csv')

        img = Image.open(img_path)

        # target 리스트 생성
        # 각 row는 [class_index, x_min, y_min, x_max, y_max] 형태가 됨
        target = []
        with open(annot_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)                    # Remove the header
            for row in csv_reader:
                # row[0]은 class label 문자열
                # row[1]~row[4]는 bbox 좌표
                #
                # 예:
                # row = ["dog", "50", "30", "200", "180"]
                #
                # 변환 후:
                # [4, 50, 30, 200, 180]
                target.append([self.label2index[row[0]]] + [int(row[i]) for i in range(1, 5)])
        # Python list를 PyTorch Tensor로 변환
        # shape: [object 개수, 5]
        # 각 object: [class_index, xmin, ymin, xmax, ymax]
        target = th.Tensor(target)

        if self.transforms is not None:
        # transform이 있으면 image와 target을 같이 변환
        # Detection에서는 image만 바꾸면 안 되고 bbox 좌표도 같이 바꿔야 함
            img, target = self.transforms((img, target))

        return img, target
