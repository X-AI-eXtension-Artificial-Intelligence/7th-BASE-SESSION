import torch as th
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import YOLOv1
from dataset import VOC_Detection
from loss import get_bb_corners, iou
from transforms import Resize, ImgToTensor

from typing import List, Tuple, Literal
import matplotlib.pyplot as plt


# =====================================
# Model Hyperparameters
# =====================================

S = 7
B = 2
D = 448


# =====================================
# DataLoader Hyperparameters
# =====================================

MINI_BATCH = 1
NUM_WORKERS = 1
PIN_MEMORY = True


# =====================================
# VOC Dataset 경로
# =====================================

PASCAL_VOC_DIR_PATH = "/media/soul/DATA/cv_datasets/PASCAL_VOC/VOC_Detection"


# =====================================
# 학습된 모델 weight 경로
# =====================================

TRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object Detection/checkpoints/trained_model_weights.pt"


# =====================================
# Device 설정
# =====================================

DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'


# =====================================
# Postprocessing Hyperparameters
# =====================================

# detection threshold
PROB_THRESHOLD = 0.005

# NMS threshold
NMS_THESHOLD = 0.6

# GT와 비교할 IOU threshold
IOU_THRESHOLD = 0.5

# confidence 계산 방식
CONF_MODE = 'class'


# AP plot 여부
PLOT = True


def get_detected_boxes(
        y: th.Tensor,
        prob_threshold: float,
        conf_mode: Literal['objectness', 'class']
) -> th.Tensor:
    """
    YOLO 출력에서 최종 bbox 추출

    과정:
        1. class probability 계산
        2. bbox confidence 계산
        3. 가장 좋은 bbox 선택
        4. threshold 이하 제거
    """

    assert conf_mode in ['objectness', 'class']

    # class score softmax
    y[..., :VOC_Detection.C] = F.softmax(
        y[..., :VOC_Detection.C],
        dim=-1
    )

    # 가장 높은 class score 선택
    class_score, class_ind = th.max(
        y[..., :VOC_Detection.C],
        dim=-1
    )

    """
    각 grid cell에서
    B개의 bbox 중 confidence 가장 높은 bbox 선택
    """

    objectness, bboxes_ind = th.max(
        y[..., [VOC_Detection.C + i * 5 for i in range(B)]],
        dim=-1
    )

    """
    선택된 bbox의 좌표 index 계산
    """

    bboxes_coords_ind = (
            th.arange(4, device=DEVICE)[None, None, None, :]
            + VOC_Detection.C
            + bboxes_ind[..., None] * 5
            + 1
    )

    # bbox 좌표 추출
    bboxes_coords = th.gather(
        y,
        dim=-1,
        index=bboxes_coords_ind
    )

    # threshold 이상 bbox만 선택
    detection_mask = (objectness > prob_threshold)

    # class index 저장
    det_class_ind = class_ind[detection_mask].reshape(-1, 1)

    # confidence 계산
    if conf_mode == 'class':

        det_conf = (
                class_score[detection_mask] *
                objectness[detection_mask]
        ).reshape(-1, 1)

    else:

        det_conf = objectness[detection_mask].reshape(-1, 1)

    # bbox를 corner format으로 변환
    bb_corners = get_bb_corners(bboxes_coords)

    # 이미지 범위 제한
    bb_corners = bb_corners.clamp(min=0, max=D)

    # mask 확장
    mask_gcs = detection_mask[..., None].expand(-1, -1, -1, 4)

    # 최종 bbox
    det_bb_corners = bb_corners[mask_gcs].reshape(-1, 4)

    # [class, confidence, xmin, ymin, xmax, ymax]
    boxes = th.cat(
        [det_class_ind, det_conf, det_bb_corners],
        dim=-1
    )

    return boxes


def non_max_suppression(
        boxes: th.Tensor,
        nms_threshold: float
) -> th.Tensor:
    """
    NMS 수행

    같은 object를 예측한 bbox 제거
    """

    nms_boxes = []

    # confidence 기준 정렬
    sort_ind = th.argsort(
        boxes[:, 1],
        descending=True
    )

    boxes = boxes[sort_ind, :]

    while len(boxes):

        # 가장 confidence 높은 bbox 선택
        box1, boxes = boxes[0], boxes[1:]

        nms_boxes.append(box1)

        box1_class = box1[0]
        box1_coords = box1[2:]

        # IOU 저장 tensor
        iou_scores = th.zeros(
            len(boxes),
            device=DEVICE
        )

        # 같은 class만 비교
        same_class = box1_class == boxes[:, 0]

        iou_scores[same_class] = iou(
            box1_coords,
            boxes[same_class][:, 2:]
        )

        # threshold 이하만 유지
        valid_boxes = iou_scores < nms_threshold

        boxes = boxes[valid_boxes]

    # 결과 정리
    if len(nms_boxes):

        boxes = th.stack(nms_boxes, dim=0)

    else:

        boxes = th.empty((0, 6), device=DEVICE)

    return boxes


def rescale_bboxes(y: th.Tensor) -> None:
    """
    YOLO bbox 좌표 복원

    YOLO 출력:
        x,y -> grid 기준 normalize
        w,h -> sqrt normalize

    이를 실제 image scale로 변환
    """

    row, col = th.meshgrid(
        th.arange(S, device=DEVICE),
        th.arange(S, device=DEVICE),
        indexing='ij'
    )

    row = row.unsqueeze(-1)
    col = col.unsqueeze(-1)

    # center 좌표 복원
    y[..., [VOC_Detection.C + i * 5 + 1 for i in range(B)]] += col
    y[..., [VOC_Detection.C + i * 5 + 2 for i in range(B)]] += row

    # image scale로 변환
    y[..., [VOC_Detection.C + i * 5 + j
            for j in [1, 2]
            for i in range(B)]] *= D / S

    # width/height 복원
    y[..., [VOC_Detection.C + i * 5 + j
            for j in [3, 4]
            for i in range(B)]] *= \
        D * y[..., [VOC_Detection.C + i * 5 + j
                    for j in [3, 4]
                    for i in range(B)]]


def postprocessing(
        y: th.Tensor,
        prob_threshold: float,
        conf_mode: Literal['objectness', 'class'],
        nms_threshold: float
) -> th.Tensor:
    """
    전체 postprocessing pipeline
    """

    # bbox scale 복원
    rescale_bboxes(y)

    # bbox 추출
    boxes = get_detected_boxes(
        y,
        prob_threshold,
        conf_mode
    )

    # NMS 적용
    boxes = non_max_suppression(
        boxes,
        nms_threshold
    )

    return boxes