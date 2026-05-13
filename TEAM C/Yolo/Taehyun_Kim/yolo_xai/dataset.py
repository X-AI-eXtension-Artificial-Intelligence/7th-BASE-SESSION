import torch as th
from torch.utils.data import Dataset
import os
import PIL.Image as Image
import csv
from typing import Callable, Optional, Tuple, Union, List


class VOC_Detection(Dataset):
    """
    PASCAL VOC 2007 + 2012 객체 탐지 데이터셋을 위한 커스텀 Dataset 클래스.
    각 레이블(클래스)에 인덱스 번호와 시각화용 색상을 할당한다.
    """

    # VOC 데이터셋의 클래스 수: 20개
    C = 20

    # 인덱스 → 클래스 이름 매핑 (순서 고정)
    index2label = [
        "person",                                          # 0: 사람
        "bird", "cat", "cow", "dog", "horse", "sheep",    # 1~6: 동물
        "aeroplane", "bicycle", "boat", "bus", "car",     # 7~11: 탈것
        "motorbike", "train",                             # 12~13: 탈것
        "bottle", "chair", "diningtable", "pottedplant",  # 14~17: 실내 물체
        "sofa", "tvmonitor"                               # 18~19: 실내 물체
    ]

    # 클래스 이름 → 인덱스 역방향 매핑 (annotation 파싱에 사용)
    label2index = {label: index for index, label in enumerate(index2label)}

    # 클래스별 시각화 색상 (HEX 코드)
    label_clrs = [
        "#ff0000",                                        # person: 빨강
        "#2e8b57", "#808000", "#800000", "#000080",       # 동물
        "#2f4f4f", "#ffa500",
        "#00ff00", "#ba55d3", "#00fa9a", "#00ffff",       # 탈것
        "#0000ff", "#f08080", "#ff00ff",
        "#1e90ff", "#ffff54", "#dda0dd", "#ff1493",       # 실내
        "#87cefa", "#ffe4c4"
    ]

    def __init__(self, root_dir: str, split: str = 'train',
                 transforms: Optional[Callable] = None) -> None:
        """
        root_dir   : 데이터셋 루트 경로. 하위에 'train/', 'test/' 디렉토리가 있어야 함.
        split      : 'train' 또는 'test'
        transforms : 이미지와 타겟에 적용할 변환 함수 (transforms.Compose 등)
        """
        assert split == 'train' or split == 'test'
        split_dir = os.path.join(root_dir, split)

        # 이미지 파일 경로: {root_dir}/{split}/images/
        self.img_dir = os.path.join(split_dir, "images")
        # 어노테이션(CSV) 파일 경로: {root_dir}/{split}/targets/
        self.annot_dir = os.path.join(split_dir, "targets")
        # 어노테이션 파일 이름에서 확장자를 제거한 ID 목록
        self.pseudonyms = [filename[:-4] for filename in os.listdir(self.annot_dir)]

        self.transforms = transforms

    def __len__(self) -> int:
        # 데이터셋의 총 이미지(샘플) 수 반환
        return len(self.pseudonyms)

    def __getitem__(self, idx: int) -> Tuple[Union[th.Tensor, Image.Image], th.Tensor]:
        """
        주어진 인덱스에 해당하는 이미지와 타겟을 반환한다.

        반환 형식:
          img    : PIL.Image (transforms 없는 경우) 또는 Tensor
          target : (N, 5) Tensor — N개 객체, 각 행: [class_id, xmin, ymin, xmax, ymax]

        어노테이션 CSV 형식:
          헤더: class, xmin, ymin, xmax, ymax
        """
        pid = self.pseudonyms[idx]
        img_path   = os.path.join(self.img_dir,   f'{pid}.jpg')
        annot_path = os.path.join(self.annot_dir, f'{pid}.csv')

        # 이미지 로드 (RGB PIL Image)
        img = Image.open(img_path)

        # CSV 어노테이션 파싱 → [class_id, xmin, ymin, xmax, ymax] 리스트
        target = []
        with open(annot_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # 헤더 행 스킵
            for row in csv_reader:
                # 클래스 이름 → 인덱스 변환 + 좌표를 정수로 변환
                target.append([self.label2index[row[0]]] + [int(row[i]) for i in range(1, 5)])
        target = th.Tensor(target)  # (N, 5) FloatTensor

        # transforms 적용 (데이터 증강, YOLO 그리드 변환 등)
        if self.transforms is not None:
            img, target = self.transforms((img, target))

        return img, target
