"""
dataset.py + transforms.py 이해용 한국어 주석 버전

핵심 역할:
- VOC 이미지와 라벨 CSV를 읽기
- bounding box 좌표를 이미지 변환에 맞게 같이 변환하기
- 최종적으로 YOLO 학습용 target tensor 만들기
"""

import torch as th
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


class VOC_Detection(Dataset):
    """
    PASCAL VOC Detection 데이터셋을 PyTorch Dataset 형태로 감싸는 클래스.

    반환 형태:
        image, target

    target의 원래 형태:
        [[class_index, xmin, ymin, xmax, ymax], ...]
    """

    C = 20

    index2label = [
        "person", "bird", "cat", "cow", "dog", "horse", "sheep",
        "aeroplane", "bicycle", "boat", "bus", "car", "motorbike",
        "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"
    ]

    label2index = {label: i for i, label in enumerate(index2label)}

    def __init__(self, root_dir, split='train', transforms=None):
        # root_dir 아래에 train/images, train/targets 또는 test/images, test/targets가 있다고 가정합니다.
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms

    def __len__(self):
        # 실제 원본에서는 target 파일 목록 개수를 반환합니다.
        pass

    def __getitem__(self, idx):
        # 1. idx에 해당하는 이미지 파일을 읽습니다.
        # 2. idx에 해당하는 CSV annotation을 읽습니다.
        # 3. 문자열 label을 숫자 class index로 바꿉니다.
        # 4. transforms가 있으면 이미지와 target을 같이 변환합니다.
        pass


class Resize:
    """
    이미지를 output_size x output_size로 바꾸고,
    bounding box 좌표도 같은 비율로 변환합니다.
    """

    def __init__(self, output_size):
        self.d = output_size

    def __call__(self, sample):
        img, target = sample
        # 원본 width, height 확인
        # img resize
        # xmin/xmax는 width 비율로 스케일링
        # ymin/ymax는 height 비율로 스케일링
        # mask는 이미지 전체 영역
        pass


class RandomScaleTranslate:
    """
    YOLO 논문/Darknet 계열에서 많이 쓰는 위치 기반 데이터 증강입니다.

    세 가지 중 하나를 랜덤으로 수행합니다.
    1. 단순 resize
    2. zoom out 후 padding
    3. zoom in 후 crop

    중요한 점:
    이미지가 바뀌면 bounding box 좌표도 반드시 같이 바꿔야 합니다.
    """

    def __call__(self, sample):
        # 랜덤값을 뽑아서 resize / zoom out / zoom in 중 하나 적용
        # 너무 작아진 bounding box는 제거
        pass


class RandomColorJitter:
    """
    이미지 색상을 HSV 공간에서 변형합니다.

    bounding box 좌표는 색상 변형과 무관하므로 그대로 유지합니다.
    padding된 영역이 있는 경우 mask 안쪽 실제 이미지 영역만 색상 변형합니다.
    """

    def __call__(self, sample):
        pass


class RandomHorizontalFlip:
    """
    이미지를 좌우반전합니다.

    박스 좌표 변환이 중요합니다.

    원래:
        xmin, xmax

    반전 후:
        new_xmin = image_width - xmax
        new_xmax = image_width - xmin
    """

    def __call__(self, sample):
        pass


class ToYOLOTensor:
    """
    마지막 변환 단계입니다.

    목표:
        기존 target: (N_objects, 5)
        YOLO target: (S, S, C + 5)

    각 grid cell의 target 구조:
        index 0       : 이 셀에 object가 있으면 1, 없으면 0
        index 1~C     : class one-hot vector
        index C+1     : cell 내부 normalized center x
        index C+2     : cell 내부 normalized center y
        index C+3     : image 전체 기준 normalized width
        index C+4     : image 전체 기준 normalized height
    """

    def __init__(self, S, C, normalize=None):
        self.S = S
        self.C = C
        self.normalize = normalize

    def __call__(self, sample):
        img, mask, target = sample

        # 이미지 크기
        w, h = img.size

        # 셀 하나의 크기
        cell_w = w / self.S
        cell_h = h / self.S

        # bounding box 중심 좌표 계산
        center_x = (target[:, 1] + target[:, 3]) / 2
        center_y = (target[:, 2] + target[:, 4]) / 2

        # 중심점이 어느 grid cell에 속하는지 계산
        center_col = th.div(center_x, cell_w, rounding_mode="trunc").long()
        center_row = th.div(center_y, cell_h, rounding_mode="trunc").long()

        # cell 내부 좌표로 정규화
        norm_center_x = (center_x % cell_w) / cell_w
        norm_center_y = (center_y % cell_h) / cell_h

        # box width, height를 이미지 전체 크기 기준으로 정규화
        box_w = (target[:, 3] - target[:, 1]) / w
        box_h = (target[:, 4] - target[:, 2]) / h

        # 최종 YOLO target tensor 생성
        yolo_target = th.zeros((self.S, self.S, self.C + 5))

        # 각 object가 속한 cell 위치에 정답 기록
        # 같은 cell에 object가 여러 개 있으면 나중 object가 덮어쓸 수 있습니다.
        label = target[:, 0].long()
        yolo_target[center_row, center_col, :] = th.cat([
            th.ones((label.shape[0], 1)),
            one_hot(label, self.C),
            norm_center_x.unsqueeze(1),
            norm_center_y.unsqueeze(1),
            box_w.unsqueeze(1),
            box_h.unsqueeze(1),
        ], dim=1)

        return img, yolo_target
