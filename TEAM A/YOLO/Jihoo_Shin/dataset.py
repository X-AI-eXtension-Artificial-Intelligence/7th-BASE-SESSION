"""
dataset.py  ─  단일 이미지 예시 데이터셋

이 파일은 실제 데이터셋 없이 바로 실행할 수 있도록
합성 이미지(또는 임의 파일)를 사용하는 Demo Dataset을 제공합니다.

구조:
  SingleImageDataset  : 이미지 1장을 N번 반복해 배치 학습을 시뮬레이션
  DemoSyntheticDataset: numpy로 무작위 생성한 이미지 사용 (OpenCV 불필요)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os

from utils import encode_target, VOC_CLASSES

IMG_SIZE = 448   # YOLOv1 입력 크기


# ─────────────────────────────────────────────────────────────
# 1) 실제 이미지 파일 기반 데이터셋
# ─────────────────────────────────────────────────────────────
class SingleImageDataset(Dataset):
    """
    이미지 한 장과 GT 박스를 받아 동일 샘플을 `repeat` 번 반복합니다.
    과적합(memorization) 학습으로 loss 수렴을 빠르게 확인할 때 사용합니다.

    gt_boxes_xyxy: [(x1, y1, x2, y2, class_id), ...]  (픽셀 좌표, 0~448 기준)
    """

    def __init__(self,
                 image_path: str,
                 gt_boxes_xyxy: list,
                 S: int = 7,
                 C: int = 20,
                 repeat: int = 100,
                 augment: bool = True):
        assert os.path.exists(image_path), f"이미지 파일을 찾을 수 없습니다: {image_path}"
        self.img_bgr   = cv2.imread(image_path)
        self.img_bgr   = cv2.resize(self.img_bgr, (IMG_SIZE, IMG_SIZE))
        self.gt_boxes  = gt_boxes_xyxy
        self.S         = S
        self.C         = C
        self.repeat    = repeat
        self.augment   = augment

        # 타겟 텐서 (S, S, C+5)
        self.target = encode_target(gt_boxes_xyxy, S=S, C=C, img_size=IMG_SIZE)

    def __len__(self):
        return self.repeat

    def __getitem__(self, idx):
        img = self.img_bgr.copy()

        if self.augment:
            img = self._augment(img)

        # BGR → RGB, HWC → CHW, [0,255] → [0,1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)   # (3, H, W)
        return img, self.target.clone()

    @staticmethod
    def _augment(img: np.ndarray) -> np.ndarray:
        """간단한 색상 지터링 augmentation"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= np.random.uniform(0.7, 1.3)   # saturation
        hsv[..., 2] *= np.random.uniform(0.7, 1.3)   # value
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ─────────────────────────────────────────────────────────────
# 2) 합성 이미지 데이터셋 (파일 없어도 동작)
# ─────────────────────────────────────────────────────────────
class DemoSyntheticDataset(Dataset):
    """
    실제 이미지 파일 없이도 동작하는 합성 데이터셋.
    지정한 gt_boxes를 포함하는 랜덤 이미지를 생성합니다.
    """

    def __init__(self,
                 gt_boxes_xyxy: list,
                 S: int = 7,
                 C: int = 20,
                 repeat: int = 100):
        self.gt_boxes = gt_boxes_xyxy
        self.S  = S
        self.C  = C
        self.repeat = repeat
        self.target = encode_target(gt_boxes_xyxy, S=S, C=C, img_size=IMG_SIZE)

    def __len__(self):
        return self.repeat

    def __getitem__(self, idx):
        # 배경: 가우시안 노이즈
        img = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)

        # GT 박스 위치에 밝은 사각형을 그려 학습 신호를 부여
        for (x1, y1, x2, y2, cls_id) in self.gt_boxes:
            color = np.array([(cls_id * 37 % 255) / 255,
                               (cls_id * 73 % 255) / 255,
                               (cls_id * 113 % 255) / 255], dtype=np.float32)
            img[int(y1):int(y2), int(x1):int(x2)] = color

        # 약간의 노이즈 추가
        img += np.random.randn(IMG_SIZE, IMG_SIZE, 3).astype(np.float32) * 0.05
        img = np.clip(img, 0, 1)

        # HWC → CHW
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        return img_tensor, self.target.clone()


# ─────────────────────────────────────────────────────────────
# 간단한 사용 예시
# ─────────────────────────────────────────────────────────────
def create_demo_dataset(image_path: str = None, repeat: int = 200):
    """
    image_path 가 주어지면 실제 이미지를, 없으면 합성 이미지를 사용합니다.

    예시 GT 박스:
      - person  (class 14): 이미지 중앙에 위치한 큰 박스
      - dog     (class 11): 좌하단 박스
    """
    gt_boxes = [
        (112, 56, 336, 392, 14),    # person
        (22,  250, 180, 420, 11),   # dog
    ]

    if image_path and os.path.exists(image_path):
        print(f"[Dataset] 실제 이미지 사용: {image_path}")
        return SingleImageDataset(image_path, gt_boxes, repeat=repeat)
    else:
        print("[Dataset] 합성 이미지 사용 (이미지 파일 없음)")
        return DemoSyntheticDataset(gt_boxes, repeat=repeat)
