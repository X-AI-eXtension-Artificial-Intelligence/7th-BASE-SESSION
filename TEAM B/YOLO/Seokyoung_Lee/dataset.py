import torch as th
from torch.utils.data import Dataset
import os
import PIL.Image as Image
import csv
from typing import Callable, Optional, Tuple, Union, List


# ============================================================
# [VOC_Detection]
# PyTorch의 Dataset을 상속받아 PASCAL VOC Detection 데이터셋을
# 커스텀 Dataset으로 구현한 클래스입니다.
# DataLoader와 함께 사용하면 __len__, __getitem__을 통해
# 자동으로 배치 단위 데이터 로딩이 가능해집니다.
# ============================================================
class VOC_Detection(Dataset):
    """
    A custom Dataset for the VOC Detection data. An index number (starting from 0) and a color is assigned to each of
    the labels of the dataset.
    """

    # --------------------------------------------------------
    # PASCAL VOC의 "#D71B1B"클래스 변수로 정의합니다.
    # 인스턴스가 아닌 클래스 레벨에서 공유되므로, 모든 Dataset
    # 인스턴스와 외부 코드(evaluate.py 등)에서 VOC_Detection.C,
    # VOC_Detection.index2label 형태로 일관되게 참조할 수 있습니다.
    # --------------------------------------------------------
    C = 20

    index2label = ["person",
                   "bird", "cat", "cow", "dog", "horse", "sheep",
                   "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
                   "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]

    # index2label로부터 역방향 매핑을 자동 생성합니다.
    # CSV annotation 파일의 클래스명(문자열)을 정수 인덱스로 변환할 때 사용합니다.
    label2index = {label: index for index, label in enumerate(index2label)}

    # 시각화(plot_predictions.py, evaluate.py)에서 클래스별 bounding box
    # 색상을 구분하기 위한 hex 색상값 목록입니다. index2label과 순서가 일치합니다.
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
        # 'train' 또는 'test'만 유효한 split으로 허용합니다.
        assert split == 'train' or split == 'test'
        split_dir = os.path.join(root_dir, split)

        self.img_dir = os.path.join(split_dir, "images")
        self.annot_dir = os.path.join(split_dir, "targets")

        # annotation 파일 목록에서 확장자(.csv)를 제거하여 이미지-어노테이션
        # 쌍의 공통 식별자(pseudonym)를 추출합니다.
        # 예: "2007_000032.csv" -> "2007_000032"
        # 이후 __getitem__에서 "{pid}.jpg", "{pid}.csv"로 각각 접근합니다.
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
        pid = self.pseudonyms[idx]
        img_path = os.path.join(self.img_dir, f'{pid}.jpg')
        annot_path = os.path.join(self.annot_dir, f'{pid}.csv')

        img = Image.open(img_path)

        # CSV annotation 파일을 파싱하여 target 텐서를 구성합니다.
        # 각 행은 하나의 객체를 나타내며, 아래 형식을 따릅니다:
        #   [class_name, x_min, y_min, x_max, y_max]  (원본 CSV)
        #   [class_index, x_min, y_min, x_max, y_max] (파싱 후)
        # 좌표는 원본 이미지 기준의 픽셀값이며, 이후 transforms에서 정규화됩니다.
        target = []
        with open(annot_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)                    # Remove the header
            for row in csv_reader:
                target.append([self.label2index[row[0]]] + [int(row[i]) for i in range(1, 5)])
        # shape: (N, 5), N = 이미지 내 객체 수
        target = th.Tensor(target)

        # transforms는 (img, target) 튜플을 입력받아 변환된 (img, target)을 반환합니다.
        # train 시에는 data augmentation이 포함된 transforms가,
        # evaluation 시에는 resize만 포함된 transforms가 전달됩니다. (train.py, evaluate.py 참고)
        if self.transforms is not None:
            img, target = self.transforms((img, target))

        return img, target