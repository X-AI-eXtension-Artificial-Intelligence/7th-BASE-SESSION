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

# ============================================================
# 하이퍼파라미터 설정
# ============================================================

S = 7    # 그리드 크기
B = 2    # 셀당 예측 박스 수
D = 448  # 입력 이미지 크기

MINI_BATCH  = 1    # 평가 시 배치 크기 1 (이미지별 처리)
NUM_WORKERS = 1
PIN_MEMORY  = True

# ── 경로 설정 ── 본인 환경에 맞게 수정 ─────────────────────────
PASCAL_VOC_DIR_PATH   = "/path/to/VOC_Detection"
TRAINED_MODEL_WEIGHTS = "/path/to/checkpoints/trained_model_weights.pt"

DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

# ── 후처리(Postprocessing) 파라미터 ──────────────────────────
PROB_THRESHOLD = 0.005  # confidence 임계값 (mAP 계산 시 ~0으로 설정해 거의 모든 박스 유지)
NMS_THESHOLD   = 0.6    # NMS IoU 임계값 (이 이상 겹치면 낮은 confidence 박스 제거)
IOU_THRESHOLD  = 0.5    # 예측이 정답과 매칭되려면 IoU가 이 값을 초과해야 함
CONF_MODE      = 'class' # 'class': conf = objectness × 클래스 확률 / 'objectness': conf = objectness만

PLOT = True  # 클래스별 AP를 바 차트로 시각화할지 여부


def get_detected_boxes(y: th.Tensor, prob_threshold: float,
                       conf_mode: Literal['objectness', 'class']) -> th.Tensor:
    """
    모델 출력에서 유효한 탐지 결과를 추출한다.

    처리 과정:
      1. 클래스 점수에 softmax 적용
      2. 각 셀에서 B개 박스 중 objectness가 가장 높은 박스 선택
      3. prob_threshold 미만인 박스 제거
      4. 신뢰도(confidence) 계산:
           - 'class' 모드:       conf = objectness × max 클래스 확률
           - 'objectness' 모드:  conf = objectness만

    반환: (N_det, 6) Tensor — [class_id, confidence, xmin, ymin, xmax, ymax]
    """
    assert conf_mode in ['objectness', 'class']

    # 클래스 확률에 softmax 적용 (합이 1이 되도록)
    y[..., :VOC_Detection.C] = F.softmax(y[..., :VOC_Detection.C], dim=-1)
    # 각 셀에서 최대 클래스 확률과 인덱스 추출
    class_score, class_ind = th.max(y[..., :VOC_Detection.C], dim=-1)

    # 각 셀의 B개 박스 중 objectness가 가장 높은 박스 선택
    objectness, bboxes_ind = th.max(y[..., [VOC_Detection.C + i * 5 for i in range(B)]], dim=-1)
    # 선택된 박스의 좌표 인덱스 계산
    bboxes_coords_ind = (th.arange(4, device=DEVICE)[None, None, None, :]
                         + VOC_Detection.C + bboxes_ind[..., None] * 5 + 1)
    bboxes_coords = th.gather(y, dim=-1, index=bboxes_coords_ind)

    # prob_threshold를 넘는 박스만 탐지 결과로 인정
    detection_mask = (objectness > prob_threshold)

    det_class_ind = class_ind[detection_mask].reshape(-1, 1)
    if conf_mode == 'class':
        # 클래스 확률 × objectness: 어떤 클래스의 객체인지 확실할수록 높음
        det_conf = (class_score[detection_mask] * objectness[detection_mask]).reshape(-1, 1)
    else:
        det_conf = objectness[detection_mask].reshape(-1, 1)

    # 박스 좌표를 코너 형식으로 변환 후 [0, D] 범위로 클램핑
    bb_corners = get_bb_corners(bboxes_coords).clamp(min=0, max=D)
    mask_gcs = detection_mask[..., None].expand(-1, -1, -1, 4)
    det_bb_corners = bb_corners[mask_gcs].reshape(-1, 4)

    boxes = th.cat([det_class_ind, det_conf, det_bb_corners], dim=-1)
    return boxes


def non_max_suppression(boxes: th.Tensor, nms_threshold: float) -> th.Tensor:
    """
    NMS(Non-Maximum Suppression): 같은 객체를 중복 탐지한 박스를 제거.

    알고리즘:
      1. confidence 내림차순 정렬
      2. 가장 높은 confidence 박스를 결과에 추가
      3. 같은 클래스이고 IoU ≥ nms_threshold인 박스를 모두 제거
      4. 남은 박스로 2~3 반복

    반환: NMS 후 남은 박스들 (confidence 내림차순 정렬 유지)
    """
    nms_boxes = []
    sort_ind = th.argsort(boxes[:, 1], descending=True)
    boxes = boxes[sort_ind, :]

    while len(boxes):
        box1, boxes = boxes[0], boxes[1:]  # 가장 높은 confidence 박스 선택
        nms_boxes.append(box1)

        box1_class, box1_coords = box1[0], box1[2:]
        iou_scores = th.zeros(len(boxes), device=DEVICE)
        same_class = box1_class == boxes[:, 0]  # 같은 클래스인 박스만 NMS 적용
        iou_scores[same_class] = iou(box1_coords, boxes[same_class][:, 2:])
        valid_boxes = iou_scores < nms_threshold  # IoU가 임계값 미만인 박스만 유지
        boxes = boxes[valid_boxes]

    if len(nms_boxes):
        boxes = th.stack(nms_boxes, dim=0)
    else:
        boxes = th.empty((0, 6), device=DEVICE)
    return boxes


def rescale_bboxes(y: th.Tensor) -> None:
    """
    모델 출력의 박스 좌표를 픽셀 단위로 역정규화.

    모델 출력 좌표 형식:
      x_center_norm : 셀 크기로 정규화된 x 중심 (0~1, 셀 기준)
      y_center_norm : 셀 크기로 정규화된 y 중심
      sqrt(w_norm)  : 이미지 크기로 정규화된 너비의 제곱근
      sqrt(h_norm)  : 이미지 크기로 정규화된 높이의 제곱근

    변환 과정:
      1. 셀 오프셋 + 셀 열/행 인덱스 → 이미지 내 절대 중심 좌표
      2. sqrt 값을 제곱해 실제 크기 복원 후 D 배율 적용
    """
    row, col = th.meshgrid(th.arange(S, device=DEVICE), th.arange(S, device=DEVICE), indexing='ij')
    row = row.unsqueeze(-1)
    col = col.unsqueeze(-1)

    # x 좌표 + 셀 열 인덱스 → 이미지 내 x 중심 (픽셀)
    y[..., [VOC_Detection.C + i * 5 + 1 for i in range(B)]] += col
    # y 좌표 + 셀 행 인덱스 → 이미지 내 y 중심 (픽셀)
    y[..., [VOC_Detection.C + i * 5 + 2 for i in range(B)]] += row
    # 셀 정규화 → 픽셀 단위 (× D/S)
    y[..., [VOC_Detection.C + i * 5 + j for j in [1, 2] for i in range(B)]] *= D / S

    # sqrt(크기) → 실제 크기^2 → 픽셀 단위: sqrt_val * sqrt_val * D
    y[..., [VOC_Detection.C + i * 5 + j for j in [3, 4] for i in range(B)]] *= D * y[
        ..., [VOC_Detection.C + i * 5 + j for j in [3, 4] for i in range(B)]
    ]


def postprocessing(y, prob_threshold, conf_mode, nms_threshold):
    """
    모델 출력 y에 세 단계 후처리 적용:
      1. rescale_bboxes   : 좌표를 픽셀 단위로 복원
      2. get_detected_boxes: 낮은 confidence 박스 제거
      3. non_max_suppression: 중복 탐지 박스 제거

    반환: (N_det, 6) — [class_id, confidence, xmin, ymin, xmax, ymax]
    """
    rescale_bboxes(y)
    boxes = get_detected_boxes(y, prob_threshold, conf_mode)
    boxes = non_max_suppression(boxes, nms_threshold)
    return boxes


def evaluate_predictions(bboxes_gt, bboxes_pred):
    """
    단일 이미지에서 예측 박스가 정답과 매칭되는지 평가.

    매칭 조건:
      - 같은 클래스
      - IoU > IOU_THRESHOLD (기본 0.5)
      - 각 정답 박스는 최대 하나의 예측 박스와만 매칭

    반환: (N_pred,) 1D Tensor — 1이면 True Positive, 0이면 False Positive
    """
    n_pred = bboxes_pred.shape[0]
    true_predictions = th.zeros(n_pred, device=DEVICE)

    for pbox_ind, pbox in enumerate(bboxes_pred):
        n_gt = bboxes_gt.shape[0]
        pbox_coords      = pbox[2:]
        bboxes_gt_coords = bboxes_gt[:, 1:]
        pbox_class_ind   = pbox[0].long()
        bboxes_gt_class_ind = bboxes_gt[:, 0].long()

        iou_scores = th.zeros(n_gt, device=DEVICE)
        same_class = pbox_class_ind == bboxes_gt_class_ind
        iou_scores[same_class] = iou(pbox_coords, bboxes_gt_coords[same_class])

        if iou_scores.shape[0]:
            best_iou, gt_box_ind = th.max(iou_scores, dim=0)
            if best_iou > IOU_THRESHOLD:
                true_predictions[pbox_ind] = 1
                # 이미 매칭된 정답 박스는 제거 (중복 매칭 방지)
                bboxes_gt = bboxes_gt[th.arange(n_gt, device=DEVICE) != gt_box_ind]

    return true_predictions


def evaluate_model(model, test_loader):
    """
    VOC 테스트셋 전체에 대해 mAP(mean Average Precision) 계산.

    AP 계산 방식 (11-point interpolation):
      1. 각 클래스별로 confidence 내림차순 정렬
      2. 누적 precision과 recall 계산
      3. 보간법 적용: p(r) = max(p(r')), r' ≥ r  (단조 감소 보장)
      4. AP = Σ p(r_k) × (r_k - r_{k-1})
      5. mAP = 모든 클래스 AP의 평균

    반환: (mAP 값, 클래스별 AP 리스트)
    """
    total_class_pred_bboxes = th.zeros(VOC_Detection.C, device=DEVICE)  # 클래스별 예측 박스 수
    total_class_gt_bboxes   = th.zeros(VOC_Detection.C, device=DEVICE)  # 클래스별 정답 박스 수
    total_predictions = th.empty((0, 3), device=DEVICE)  # [class_id, confidence, is_tp]

    with th.no_grad():
        model.eval()
        for x, bboxes_gt in test_loader:
            x, bboxes_gt = x.to(DEVICE), bboxes_gt.to(DEVICE).squeeze(0)
            y = model(x)
            bboxes_pred = postprocessing(y, prob_threshold=PROB_THRESHOLD,
                                         conf_mode=CONF_MODE, nms_threshold=NMS_THESHOLD)

            # 클래스별 예측/정답 박스 수 집계
            total_class_pred_bboxes += th.bincount(bboxes_pred[:, 0].long(), minlength=VOC_Detection.C)
            total_class_gt_bboxes   += th.bincount(bboxes_gt[:, 0].long(),   minlength=VOC_Detection.C)

            # TP/FP 판정
            is_true_pred_bbox = evaluate_predictions(bboxes_gt, bboxes_pred)
            sample_predictions = th.stack(
                [bboxes_pred[:, 0], bboxes_pred[:, 1], is_true_pred_bbox], dim=-1
            )
            total_predictions = th.cat([total_predictions, sample_predictions])

    # 클래스별 AP 계산
    average_precisions = []
    for c in range(VOC_Detection.C):
        class_mask = total_predictions[:, 0] == c
        if not th.max(class_mask):
            continue

        class_predictions = total_predictions[class_mask]
        sort_ind  = th.argsort(class_predictions[:, 1], descending=True)
        sorted_tp = class_predictions[sort_ind, 2]
        cumsum_tp = th.cumsum(sorted_tp, dim=0)

        # Precision 계산 후 보간(단조 감소 보장)
        class_precision = cumsum_tp / th.arange(start=1, end=total_class_pred_bboxes[c] + 1, device=DEVICE)
        class_precision = th.flip(th.cummax(th.flip(class_precision, [0]), dim=0)[0], [0])

        # Recall 계산 (시작점 0 추가)
        class_recall = cumsum_tp / total_class_gt_bboxes[c]
        class_recall = th.cat([th.zeros(1, device=DEVICE), class_recall], dim=0)

        # AP = 면적 합산 (Recall 증가량 × Precision)
        class_ap = th.sum(class_precision * (class_recall[1:] - class_recall[:-1]))
        average_precisions.append(class_ap.item() * 100)

    mAP = sum(average_precisions) / len(average_precisions)
    return mAP, average_precisions


def setup_evaluation():
    """
    평가를 위한 모델과 테스트 데이터로더 초기화.
    학습된 가중치를 체크포인트에서 로드.
    """
    model = YOLOv1(S=S, B=B, C=VOC_Detection.C).to(DEVICE)
    trained_model_weights = th.load(TRAINED_MODEL_WEIGHTS)
    model.load_state_dict(trained_model_weights)

    test_dataset = VOC_Detection(
        root_dir=PASCAL_VOC_DIR_PATH, split='test',
        transforms=transforms.Compose([
            Resize(output_size=D),
            ImgToTensor(normalize=[[0.4549, 0.4341, 0.4010], [0.2703, 0.2672, 0.2808]])
        ])
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=MINI_BATCH,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return model, test_loader


def plot_class_ap(average_precisions: List[float]) -> None:
    """
    각 클래스의 AP를 수평 막대 차트로 시각화.
    막대 색상은 dataset.py의 label_clrs에 정의된 클래스별 색상 사용.
    """
    fig, ax = plt.subplots()
    bars = ax.barh(VOC_Detection.index2label, average_precisions, color=VOC_Detection.label_clrs)
    ax.bar_label(bars, labels=[f'{ap:.1f}%' for ap in average_precisions])
    ax.invert_yaxis()  # 위에서 아래로 클래스 순서

    # 테두리 제거 (깔끔한 시각화)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.yaxis.set_ticks_position('none')

    ax.set_title('Class Average Precisions')
    plt.show()


def main():
    model, test_loader = setup_evaluation()
    mAP, average_precisions = evaluate_model(model, test_loader)
    print(f'Mean Average Precision = {mAP:.1f}%')

    if PLOT:
        plot_class_ap(average_precisions)


if __name__ == '__main__':
    main()
