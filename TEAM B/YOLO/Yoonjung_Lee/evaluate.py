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

# ... 하이퍼파라미터 설정 생략 ...

def get_detected_boxes(y: th.Tensor, prob_threshold: float, conf_mode: Literal['objectness', 'class']) -> th.Tensor:
    """
    모델 출력 텐서에서 유효한 경계 상자 정보를 추출하는 과정이다.
    """
    # 클래스 확률에 Softmax를 적용하여 확률 분포로 변환하는 단계
    y[..., :VOC_Detection.C] = F.softmax(y[..., :VOC_Detection.C], dim=-1)
    # 각 격자 셀에서 가장 높은 확률을 가진 클래스와 그 인덱스 추출 수행
    class_score, class_ind = th.max(y[..., :VOC_Detection.C], dim=-1)
    # B개의 상자 중 신뢰도(Objectness)가 가장 높은 상자 선택 과정
    objectness, bboxes_ind = th.max(y[..., [VOC_Detection.C + i * 5 for i in range(B)]], dim=-1)
    
    # 선택된 상자의 좌표 데이터 인덱스 계산 및 해당 좌표값 추출 작업
    bboxes_coords_ind = th.arange(4, device=DEVICE)[None, None, None, :] + VOC_Detection.C + bboxes_ind[..., None] * 5 + 1
    bboxes_coords = th.gather(y, dim=-1, index=bboxes_coords_ind)
    
    # 설정된 임계값(prob_threshold)보다 낮은 신뢰도를 가진 상자 필터링 마스크 생성
    detection_mask = (objectness > prob_threshold)
    det_class_ind = class_ind[detection_mask].reshape(-1, 1)
    
    # 신뢰도 모드(class/objectness)에 따른 최종 점수 계산 수행
    if conf_mode == 'class':
        det_conf = (class_score[detection_mask] * objectness[detection_mask]).reshape(-1, 1)
    else:
        det_conf = objectness[detection_mask].reshape(-1, 1)

    # 중심점 기반 좌표를 꼭짓점 좌표로 변환 및 이미지 경계 내로 제한(Clamp) 처리
    bb_corners = get_bb_corners(bboxes_coords).clamp(min=0, max=D)
    det_bb_corners = bb_corners[detection_mask[..., None].expand(-1, -1, -1, 4)].reshape(-1, 4)

    # [클래스ID, 신뢰도점수, xmin, ymin, xmax, ymax] 형태의 최종 박스 리스트 생성
    return th.cat([det_class_ind, det_conf, det_bb_corners], dim=-1)

def non_max_suppression(boxes: th.Tensor, nms_threshold: float) -> th.Tensor:
    """
    중복된 탐지 상자들을 정리하여 가장 확실한 하나만 남기는 NMS 알고리즘 수행 과정이다.
    """
    # 신뢰도 점수 기준 내림차순 정렬 수행
    sort_ind = th.argsort(boxes[:, 1], descending=True)
    boxes = boxes[sort_ind, :]
    nms_boxes = []

    while len(boxes):
        # 점수가 가장 높은 상자를 선택하고 결과 리스트에 추가하는 작업
        box1, boxes = boxes[0], boxes[1:]
        nms_boxes.append(box1)

        if not len(boxes): break
        
        # 선택된 상자와 나머지 상자들 간의 IOU 계산 수행
        iou_scores = th.zeros(len(boxes), device=DEVICE)
        same_class = box1[0] == boxes[:, 0] # 같은 클래스인 경우만 비교
        iou_scores[same_class] = iou(box1[2:], boxes[same_class][:, 2:])
        
        # IOU가 임계값(nms_threshold)보다 높은 중복 상자 제거(Discard) 처리
        valid_boxes = iou_scores < nms_threshold
        boxes = boxes[valid_boxes]

    return th.stack(nms_boxes, dim=0) if len(nms_boxes) else th.empty((0, 6), device=DEVICE)


def rescale_bboxes(y: th.Tensor) -> None:
    """
    격자별 상대 좌표와 루트 크기 정보를 실제 픽셀 좌표로 복원하는 과정이다.
    """
    # 각 격자 셀의 위치(row, col) 오프셋 정보 생성
    row, col = th.meshgrid(th.arange(S, device=DEVICE), th.arange(S, device=DEVICE), indexing='ij')
    
    # 1. 중심점(x, y) 복원: 격자 오프셋을 더하고 전체 해상도 대비 격자 크기 비율 반영 수행
    y[..., [VOC_Detection.C + i * 5 + 1 for i in range(B)]] += col.unsqueeze(-1)
    y[..., [VOC_Detection.C + i * 5 + 2 for i in range(B)]] += row.unsqueeze(-1)
    y[..., [VOC_Detection.C + i * 5 + j for j in [1, 2] for i in range(B)]] *= D / S

    # 2. 크기(w, h) 복원: 루트를 제거하기 위한 제곱 연산 및 전체 해상도 곱셈 처리
    y[..., [VOC_Detection.C + i * 5 + j for j in [3, 4] for i in range(B)]] *= D * y[..., [VOC_Detection.C + i * 5 + j for j in [3, 4] for i in range(B)]]

def evaluate_model(model: YOLOv1, test_loader: DataLoader) -> Tuple[float, List[float]]:
    """
    테스트 데이터셋 전체에 대해 mAP(Mean Average Precision)를 측정하는 성능 평가 과정이다.
    """
    total_predictions = th.empty((0, 3), device=DEVICE)
    # ... 전체 예측값 수집 과정 생략 ...

    average_precisions = []
    for c in range(VOC_Detection.C):
        # 클래스별로 예측 성공 여부(TP/FP)와 신뢰도를 정렬하여 정밀도-재현율 곡선 계산 수행
        # Precision Interpolation 및 면적(AP) 계산을 통한 최종 성적 산출 작업
        # ... AP 계산 로직 수행 ...
        average_precisions.append(class_ap.item() * 100)

    # 모든 클래스의 AP 평균값인 mAP 반환
    return sum(average_precisions) / len(average_precisions), average_precisions