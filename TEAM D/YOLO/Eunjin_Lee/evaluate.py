"""
evaluate.py - YOLOv1 모델 평가 (mAP 계산)

학습된 YOLOv1 모델의 객체탐지 성능을 Mean Average Precision (mAP)으로 평가합니다.

후처리 파이프라인:
  1. 바운딩 박스 좌표 역정규화 (rescale_bboxes)
  2. 각 셀에서 최적 박스 선택 + 신뢰도 계산 (get_detected_boxes)
  3. Non-Maximum Suppression (non_max_suppression)
  4. GT와 매칭하여 TP/FP 판정 (evaluate_predictions)
  5. 클래스별 AP 계산 → mAP (evaluate_model)
"""

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

# ===== 모델 하이퍼파라미터 =====
S = 7       # 그리드 크기
B = 2       # 셀당 바운딩 박스 수
D = 448     # 입력 이미지 크기

# ===== 데이터 로딩 하이퍼파라미터 =====
MINI_BATCH = 1      # 배치 크기 (평가 시 1장씩)
NUM_WORKERS = 1
PIN_MEMORY = True

# ===== 데이터셋/모델 경로 =====
PASCAL_VOC_DIR_PATH = "/media/soul/DATA/cv_datasets/PASCAL_VOC/VOC_Detection"
TRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                        "Detection/checkpoints/trained_model_weights.pt"

# ===== 디바이스 설정 =====
DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

# ===== 후처리 하이퍼파라미터 =====
PROB_THRESHOLD = 0.005  # 신뢰도 임계값 (mAP 계산용으로 낮게 설정)
NMS_THESHOLD = 0.6      # NMS IoU 임계값
IOU_THRESHOLD = 0.5     # TP 판정 IoU 임계값
CONF_MODE = 'class'     # 신뢰도 모드: 'class' 또는 'objectness'

# ===== 시각화 설정 =====
PLOT = True  # 클래스별 AP 그래프 표시 여부


def get_detected_boxes(y: th.Tensor,
                       prob_threshold: float,
                       conf_mode: Literal['objectness', 'class']) -> th.Tensor:
    """
    모델 출력에서 탐지된 바운딩 박스를 추출합니다.

    각 셀의 B개 박스 중 가장 높은 예측 IoU를 가진 박스만 선택하고,
    임계값 이하의 박스는 제거합니다.

    신뢰도 계산:
      - 'class' 모드: predicted_IoU × class_probability (softmax 적용)
      - 'objectness' 모드: predicted_IoU만 사용

    :param y: 모델 출력 (좌표가 코너 형식으로 변환된 상태)
    :param prob_threshold: 신뢰도 임계값
    :param conf_mode: 신뢰도 계산 모드
    :return: 탐지된 박스 텐서 (N, 6): [클래스, 신뢰도, xmin, ymin, xmax, ymax]
    """
    assert conf_mode in ['objectness', 'class']

    # 클래스 확률에 softmax 적용
    y[..., :VOC_Detection.C] = F.softmax(y[..., :VOC_Detection.C], dim=-1)
    class_score, class_ind = th.max(y[..., :VOC_Detection.C], dim=-1)

    # B개 박스 중 최고 objectness를 가진 박스 선택
    objectness, bboxes_ind = th.max(y[..., [VOC_Detection.C + i * 5 for i in range(B)]], dim=-1)
    bboxes_coords_ind = th.arange(4, device=DEVICE)[None, None, None, :] + VOC_Detection.C + bboxes_ind[
        ..., None] * 5 + 1
    bboxes_coords = th.gather(y, dim=-1, index=bboxes_coords_ind)

    # 임계값 이상인 박스만 선택
    detection_mask = (objectness > prob_threshold)

    det_class_ind = class_ind[detection_mask].reshape(-1, 1)
    if conf_mode == 'class':
        det_conf = (class_score[detection_mask] * objectness[detection_mask]).reshape(-1, 1)
    else:
        det_conf = objectness[detection_mask].reshape(-1, 1)

    # 좌표를 [0, D] 범위로 클램핑
    bb_corners = get_bb_corners(bboxes_coords).clamp(min=0, max=D)
    mask_gcs = detection_mask[..., None].expand(-1, -1, -1, 4)
    det_bb_corners = bb_corners[mask_gcs].reshape(-1, 4)

    boxes = th.cat([det_class_ind, det_conf, det_bb_corners], dim=-1)
    return boxes


def non_max_suppression(boxes: th.Tensor,
                        nms_threshold: float) -> th.Tensor:
    """
    Non-Maximum Suppression (NMS).
    같은 클래스의 겹치는 박스 중 신뢰도가 가장 높은 박스만 남깁니다.

    알고리즘:
    1. 신뢰도 내림차순 정렬
    2. 가장 높은 신뢰도 박스를 선택
    3. 같은 클래스이고 IoU ≥ nms_threshold인 박스 제거
    4. 남은 박스에 대해 2-3 반복

    :param boxes: 탐지된 박스 (N, 6): [클래스, 신뢰도, xmin, ymin, xmax, ymax]
    :param nms_threshold: NMS IoU 임계값
    :return: NMS 후 남은 박스
    """
    nms_boxes = []
    sort_ind = th.argsort(boxes[:, 1], descending=True)
    boxes = boxes[sort_ind, :]
    while len(boxes):
        box1, boxes = boxes[0], boxes[1:]
        nms_boxes.append(box1)

        box1_class, box1_coords = box1[0], box1[2:]
        iou_scores = th.zeros(len(boxes), device=DEVICE)
        same_class = box1_class == boxes[:, 0]
        iou_scores[same_class] = iou(box1_coords, boxes[same_class][:, 2:])
        # IoU < 임계값인 박스만 유지
        valid_boxes = iou_scores < nms_threshold
        boxes = boxes[valid_boxes]

    if len(nms_boxes):
        boxes = th.stack(nms_boxes, dim=0)
    else:
        boxes = th.empty((0, 6), device=DEVICE)
    return boxes


def rescale_bboxes(y: th.Tensor) -> None:
    """
    모델 출력의 정규화된 좌표를 원본 이미지 스케일로 역변환합니다.

    모델 출력 형식: (x_norm, y_norm, √w_norm, √h_norm)
      - x_norm, y_norm: 셀 크기로 정규화된 중심 좌표
      - √w_norm, √h_norm: 이미지 크기로 정규화된 크기의 제곱근

    변환 후: (x_pixel, y_pixel, w_pixel, h_pixel)

    :param y: 모델 예측 텐서 (in-place 수정)
    """
    row, col = th.meshgrid(th.arange(S, device=DEVICE), th.arange(S, device=DEVICE), indexing='ij')
    row = row.unsqueeze(-1)
    col = col.unsqueeze(-1)

    # 중심 좌표: 셀 오프셋 추가 후 픽셀 스케일로 변환
    y[..., [VOC_Detection.C + i * 5 + 1 for i in range(B)]] += col
    y[..., [VOC_Detection.C + i * 5 + 2 for i in range(B)]] += row
    y[..., [VOC_Detection.C + i * 5 + j for j in [1, 2] for i in range(B)]] *= D / S

    # 크기: √w × √w = w (제곱으로 복원) 후 이미지 크기 곱
    y[..., [VOC_Detection.C + i * 5 + j for j in [3, 4] for i in range(B)]] *= D * y[..., [VOC_Detection.C + i * 5 + j
                                                                                           for j in [3, 4] for i in
                                                                                           range(B)]]


def postprocessing(y: th.Tensor,
                   prob_threshold: float,
                   conf_mode: Literal['objectness', 'class'],
                   nms_threshold: float) -> th.Tensor:
    """
    모델 출력에 대한 전체 후처리 파이프라인.
    좌표 역정규화 → 박스 선택 → NMS

    :param y: 모델 출력 (N, S, S, C+B*5)
    :param prob_threshold: 신뢰도 임계값
    :param conf_mode: 신뢰도 계산 모드
    :param nms_threshold: NMS IoU 임계값
    :return: 최종 탐지 결과 (M, 6): [클래스, 신뢰도, xmin, ymin, xmax, ymax]
    """
    rescale_bboxes(y)
    boxes = get_detected_boxes(y, prob_threshold, conf_mode)
    boxes = non_max_suppression(boxes, nms_threshold)
    return boxes


def evaluate_predictions(bboxes_gt: th.Tensor, bboxes_pred: th.Tensor) -> th.Tensor:
    """
    단일 이미지에 대해 예측 박스와 GT 박스를 매칭하여 TP/FP를 판정합니다.

    매칭 조건:
      - 같은 클래스
      - IoU > IOU_THRESHOLD
      - 각 GT 박스는 하나의 예측 박스에만 매칭 가능

    신뢰도가 높은 예측 박스부터 우선 매칭합니다 (NMS 후 이미 정렬됨).

    :param bboxes_gt: GT 박스 (N, 5): [클래스, xmin, ymin, xmax, ymax]
    :param bboxes_pred: 예측 박스 (M, 6): [클래스, 신뢰도, xmin, ymin, xmax, ymax]
    :return: TP 마스크 (M,): 매칭 성공=1, 실패=0
    """
    n_pred = bboxes_pred.shape[0]
    true_predictions = th.zeros(n_pred, device=DEVICE)

    for pbox_ind, pbox in enumerate(bboxes_pred):
        n_gt = bboxes_gt.shape[0]
        pbox_coords = pbox[2:]
        bboxes_gt_coords = bboxes_gt[:, 1:]
        pbox_class_ind = pbox[0].long()
        bboxes_gt_class_ind = bboxes_gt[:, 0].long()

        # 같은 클래스인 GT 박스와만 IoU 계산
        iou_scores = th.zeros(n_gt, device=DEVICE)
        same_class = pbox_class_ind == bboxes_gt_class_ind
        iou_scores[same_class] = iou(pbox_coords, bboxes_gt_coords[same_class])

        if iou_scores.shape[0]:
            best_iou, gt_box_ind = th.max(iou_scores, dim=0)
            if best_iou > IOU_THRESHOLD:
                true_predictions[pbox_ind] = 1
                # 매칭된 GT 박스 제거 (중복 매칭 방지)
                bboxes_gt = bboxes_gt[th.arange(n_gt, device=DEVICE) != gt_box_ind]

    return true_predictions


def evaluate_model(model: YOLOv1, test_loader: DataLoader) -> Tuple[float, List[float]]:
    """
    테스트 세트에서 mAP(Mean Average Precision)를 계산합니다.

    계산 과정:
    1. 모든 테스트 이미지에 대해 예측 수행 및 TP/FP 판정
    2. 클래스별로 신뢰도 내림차순 정렬
    3. Precision-Recall 곡선 계산
    4. Precision 보간 (monotone decreasing)
    5. AP = ∫ precision d(recall)
    6. mAP = 클래스별 AP의 평균

    :param model: 학습된 YOLOv1 모델
    :param test_loader: 테스트 데이터 로더
    :return: (mAP, 클래스별 AP 리스트)
    """
    total_class_pred_bboxes = th.zeros(VOC_Detection.C, device=DEVICE)
    total_class_gt_bboxes = th.zeros(VOC_Detection.C, device=DEVICE)
    total_predictions = th.empty((0, 3), device=DEVICE)

    with th.no_grad():
        model.eval()
        for x, bboxes_gt in test_loader:
            x, bboxes_gt = x.to(DEVICE), bboxes_gt.to(DEVICE).squeeze(0)
            y = model(x)
            bboxes_pred = postprocessing(y,
                                         prob_threshold=PROB_THRESHOLD,
                                         conf_mode=CONF_MODE,
                                         nms_threshold=NMS_THESHOLD)

            # 클래스별 예측/GT 박스 수 누적
            total_class_pred_bboxes += th.bincount(bboxes_pred[:, 0].long(), minlength=VOC_Detection.C)
            total_class_gt_bboxes += th.bincount(bboxes_gt[:, 0].long(), minlength=VOC_Detection.C)

            # 예측 결과 저장: [클래스, 신뢰도, TP여부]
            predictions_class_ind = bboxes_pred[:, 0]
            predictions_conf = bboxes_pred[:, 1]
            is_true_pred_bbox = evaluate_predictions(bboxes_gt, bboxes_pred)

            sample_predictions = th.stack([predictions_class_ind, predictions_conf, is_true_pred_bbox], dim=-1)
            total_predictions = th.cat([total_predictions, sample_predictions])

    # 클래스별 Average Precision 계산
    average_precisions = []
    for c in range(VOC_Detection.C):

        class_mask = total_predictions[:, 0] == c
        if not th.max(class_mask):
            continue

        class_predictions = total_predictions[class_mask]
        # 신뢰도 내림차순 정렬
        sort_ind = th.argsort(class_predictions[:, 1], descending=True)
        sorted_tp = class_predictions[sort_ind, 2]
        cumsum_tp = th.cumsum(sorted_tp, dim=0)

        # Precision = TP누적 / 예측수누적
        class_precision = cumsum_tp / th.arange(start=1, end=total_class_pred_bboxes[c] + 1, device=DEVICE)
        # Precision 보간: 뒤에서부터 cummax로 단조감소 보장
        class_precision = th.flip(th.cummax(th.flip(class_precision, [0]), dim=0)[0], [0])

        # Recall = TP누적 / GT총수
        class_recall = cumsum_tp / total_class_gt_bboxes[c]
        class_recall = th.cat([th.zeros(1, device=DEVICE), class_recall], dim=0)

        # AP = Σ precision × Δrecall
        class_ap = th.sum(class_precision * (class_recall[1:] - class_recall[:-1]))
        average_precisions.append(class_ap.item() * 100)

    mAP = sum(average_precisions) / len(average_precisions)
    return mAP, average_precisions


def setup_evaluation() -> Tuple[YOLOv1, DataLoader]:
    """
    평가에 필요한 모델과 데이터 로더를 초기화합니다.
    학습된 가중치를 로드하고, 테스트 데이터셋을 준비합니다.

    :return: (YOLOv1 모델, 테스트 데이터 로더)
    """
    model = YOLOv1(S=S,
                   B=B,
                   C=VOC_Detection.C).to(DEVICE)
    trained_model_weights = th.load(TRAINED_MODEL_WEIGHTS)
    model.load_state_dict(trained_model_weights)

    test_dataset = VOC_Detection(root_dir=PASCAL_VOC_DIR_PATH,
                                 split='test',
                                 transforms=transforms.Compose([
                                     Resize(output_size=D),
                                     ImgToTensor(normalize=[[0.4549, 0.4341, 0.4010],
                                                            [0.2703, 0.2672, 0.2808]])
                                 ]))

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=MINI_BATCH,
                             num_workers=NUM_WORKERS,
                             pin_memory=PIN_MEMORY)

    return model, test_loader


def plot_class_ap(average_precisions: List[float]) -> None:
    """
    클래스별 Average Precision을 수평 막대 그래프로 시각화합니다.

    :param average_precisions: 클래스별 AP 리스트 (%)
    """
    fig, ax = plt.subplots()
    bars = ax.barh(VOC_Detection.index2label, average_precisions, color=VOC_Detection.label_clrs)
    ax.bar_label(bars, labels=[f'{ap:.1f}%' for ap in average_precisions])
    ax.invert_yaxis()

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.axes.get_xaxis().set_visible(False)
    ax.yaxis.set_ticks_position('none')

    ax.set_title('Class Average Precisions')
    plt.show()


def main():
    """메인 함수: 모델 로드 → mAP 평가 → (선택적) 시각화"""
    model, test_loader = setup_evaluation()
    mAP, average_precisions = evaluate_model(model, test_loader)
    print(f'Mean Average Precision = {mAP:.1f}%')

    if PLOT:
        plot_class_ap(average_precisions)


if __name__ == '__main__':
    main()
