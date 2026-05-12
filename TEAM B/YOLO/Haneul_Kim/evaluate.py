"""
학습된 YOLOv1 모델의 성능을 평가하는 파일입니다.
모델 출력값을 실제 bounding box 좌표로 변환하고,
confidence threshold, NMS, IoU matching을 거쳐 mAP를 계산합니다.
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

# Model Hyperparameters
# YOLOv1은 이미지를 S x S grid로 나눕니
S = 7
# 각 grid cell마다 예측하는 bounding box 개수
B = 2
# 모델 입력 이미지 크기 즉 448 x 448로 resize
D = 448

# Data Loading Hyperparameters
MINI_BATCH = 1
NUM_WORKERS = 1
PIN_MEMORY = True

# VOC Dataset Directory
PASCAL_VOC_DIR_PATH = "/media/soul/DATA/cv_datasets/PASCAL_VOC/VOC_Detection"

# Trained Model Path
TRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                        "Detection/checkpoints/trained_model_weights.pt"

# Compute Device (use a GPU if available)
# GPU가 있으면 CUDA를 사용하고, 없으면 CPU를 사용
DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

# ===================== 후처리/NMS/mAP 평가 설정 =====================
# Postprocessing Hyperparameters
PROB_THRESHOLD = 0.005  # this value is set ~= 0 for the map metric calculation
NMS_THESHOLD = 0.6
IOU_THRESHOLD = 0.5
CONF_MODE = 'class'

# Plot the classes' APs
PLOT = True


# 모델 출력에서 threshold를 통과한 bounding box만 추려내는 함수
def get_detected_boxes(y: th.Tensor,
                       prob_threshold: float,
                       conf_mode: Literal['objectness', 'class']) -> th.Tensor:
    """
    For each of the B bounding boxes of each grid cell, only the box with the highest predicted IOU is considered. If
    this box has a predicted IOU less than or equal to the prob_threshold, the box is discarded. For each box, the
    confidence is defined as:
        - predicted IOU * class probability, for the 'class' confidence mode, and
        - predicted IOU, for the 'objectness' mode
    To scale the class probabilities appropriately, the softmax function is used.

    :param y: The output of the YOLO v1 model with bounding boxes coordinates rescaled to a corners-format
    :param prob_threshold: The probability threshold under which the bounding boxes are discarded.
    :param conf_mode: The confidence mode determines how the confidence score of each predicted bounding box is
                      calculated.
    :return: A 2D tensor whose rows correspond to detected bounding boxes. For each bounding box, the class, the
             confidence and the corner coordinates (xmin, xmax, ymin, ymax) are listed.
    """

 # confidence 계산 방식이 올바른지 확인
    assert conf_mode in ['objectness', 'class']

 # class score를 확률처럼 해석하기 위해 softmax를 적용
    y[..., :VOC_Detection.C] = F.softmax(y[..., :VOC_Detection.C], dim=-1)
 # 각 grid cell에서 가장 확률이 높은 class를 선택
    class_score, class_ind = th.max(y[..., :VOC_Detection.C], dim=-1)
 # B개의 box 중 objectness가 가장 높은 box를 선택
    objectness, bboxes_ind = th.max(y[..., [VOC_Detection.C + i * 5 for i in range(B)]], dim=-1)
    bboxes_coords_ind = th.arange(4, device=DEVICE)[None, None, None, :] + VOC_Detection.C + bboxes_ind[
        ..., None] * 5 + 1
    bboxes_coords = th.gather(y, dim=-1, index=bboxes_coords_ind)
 # threshold보다 confidence가 낮은 box는 제거
    detection_mask = (objectness > prob_threshold)

    det_class_ind = class_ind[detection_mask].reshape(-1, 1)
    if conf_mode == 'class':
        det_conf = (class_score[detection_mask] * objectness[detection_mask]).reshape(-1, 1)
    else:
        det_conf = objectness[detection_mask].reshape(-1, 1)

    bb_corners = get_bb_corners(bboxes_coords).clamp(min=0, max=D)
    mask_gcs = detection_mask[..., None].expand(-1, -1, -1, 4)
    det_bb_corners = bb_corners[mask_gcs].reshape(-1, 4)

    boxes = th.cat([det_class_ind, det_conf, det_bb_corners], dim=-1)
    return boxes


# 같은 객체를 여러 box가 중복 검출했을 때 가장 좋은 box만 남기는 NMS 함수
def non_max_suppression(boxes: th.Tensor,
                        nms_threshold: float) -> th.Tensor:
    """
    Some bounding boxes predictions of the YOLO model (e.g. predictions of neighbouring grid cells) may detect the same
    object. When two bounding boxes have the same class prediction and their IOU score is greater than or equal to the
    nms_threshold, only the bounding box with the highest confidence is considered to be valid, while the other bounding
    box is discarded.

    :param boxes: The predicted bounding boxes for a single image. For each box, the class, the confidence and the
                  corner coordinates are listed.
    :param nms_threshold: The non-max suppression threshold over which overlapping bounding boxes of the same class are
                          discarded.
    :return: The predicted boxes for this image after the non-max suppression operation.
    """
    nms_boxes = []
 # confidence가 높은 box부터 처리하기 위해 내림차순 정렬
    sort_ind = th.argsort(boxes[:, 1], descending=True)
    boxes = boxes[sort_ind, :]
 # 가장 confidence가 높은 box를 하나 선택하고, 겹치는 box들을 제거하는 과정을 반복
    while len(boxes):
        box1, boxes = boxes[0], boxes[1:]
        nms_boxes.append(box1)

        box1_class, box1_coords = box1[0], box1[2:]
        iou_scores = th.zeros(len(boxes), device=DEVICE)
 # 같은 class끼리만 NMS를 적용
        same_class = box1_class == boxes[:, 0]
        iou_scores[same_class] = iou(box1_coords, boxes[same_class][:, 2:])
 # IoU가 threshold 이상이면 같은 객체로 보고 제거
        valid_boxes = iou_scores < nms_threshold
        boxes = boxes[valid_boxes]

    if len(nms_boxes):
        boxes = th.stack(nms_boxes, dim=0)
    else:
        boxes = th.empty((0, 6), device=DEVICE)
    return boxes


# YOLO 형식의 좌표를 실제 이미지 픽셀 좌표로 되돌리는 함수
def rescale_bboxes(y: th.Tensor) -> None:
    """
    The coordinates of the predicted bounding boxes are transformed. Each of the predicted bounding boxes of the YOLO
    model are represented as (x_center_norm, y_center_norm, sqrt(width_norm), sqrt(height_norm)). The predicted center
    coordinates x_center_norm and y_center_norm are normalized by the dimension of the grid cells, while the width_norm
    and height_norm are normalized by the dimension of the resized image. The function rescales these coordinates to
    the (x_center, y_center, width, height) by reversing the normalizations.

    :param y: The predictions of the YOLOv1 model
    """
 # 각 grid cell의 row/column 위치를 만들어 center 좌표 복원에 사용
    row, col = th.meshgrid(th.arange(S, device=DEVICE), th.arange(S, device=DEVICE), indexing='ij')
    row = row.unsqueeze(-1)
    col = col.unsqueeze(-1)

    y[..., [VOC_Detection.C + i * 5 + 1 for i in range(B)]] += col
    y[..., [VOC_Detection.C + i * 5 + 2 for i in range(B)]] += row
    y[..., [VOC_Detection.C + i * 5 + j for j in [1, 2] for i in range(B)]] *= D / S

    y[..., [VOC_Detection.C + i * 5 + j for j in [3, 4] for i in range(B)]] *= D * y[..., [VOC_Detection.C + i * 5 + j
                                                                                           for j in [3, 4] for i in
                                                                                           range(B)]]


# 모델 출력 후처리 전체 과정을 묶은 함수: 좌표 복원 → box 선택 → NMS.
def postprocessing(y: th.Tensor,
                   prob_threshold: float,
                   conf_mode: Literal['objectness', 'class'],
                   nms_threshold: float) -> th.Tensor:
    """
    Rescale the bounding boxes coordinates to the corner-format. Select the best of the B boxes for each grid cell and
    calculate the confidence for this box. Discard the bounding boxes with a low predicted IOU and those that overlap
    significantly with other higher-confidence boxes.

    :param y: The output the YOLOv1 model.
    :param prob_threshold: The probability threshold under which the bounding boxes are discarded.
    :param conf_mode: The confidence mode determines how the confidence score of each predicted bounding box is
                      calculated.
    :param nms_threshold: The non-max suppression threshold over which overlapping bounding boxes of the same class are
                          discarded.
    :return: A tensor with a row for each bounding box. Each row contains the box's assigned class, its confidence and
             the predicted coordinates xmin, ymin, xmax, ymax.
    """
    rescale_bboxes(y)
    boxes = get_detected_boxes(y, prob_threshold, conf_mode)
    boxes = non_max_suppression(boxes, nms_threshold)
    return boxes


# 예측 box가 실제 정답 box와 매칭되는지 판단하는 함수
def evaluate_predictions(bboxes_gt: th.Tensor, bboxes_pred: th.Tensor) -> th.Tensor:
    """
    This function evaluates whether the predicted bounding boxes match a ground truth box for a single image. Each
    ground truth box can be matched to a single predicted bounding box. For a predicted and a ground truth bounding box
    to match, their IOU score must be greater than the IOU_THRESHOLD and the object's class must be the same. The
    predicted bounding boxes with the highest confidence values are matched first.

    NOTE: The predicted bounding boxes are already sorted in descending order with respect to their confidence values
    after the non-max suppression operation.

    :param bboxes_gt: The ground truth bounding boxes of the image. For each object/row, the following
                      attributes/columns are listed: class, xmin, ymin, xmax, ymax
    :param bboxes_pred: The predicted bounding boxes of the image. For each object/row, the following
                        attributes/columns are listed: class, confidence, xmin, ymin, xmax, ymax
    :return: A 1D tensor with length equal to the number of the predicted bounding boxes. If the bounding box i matches
             a ground truth box, then true_predictions[i]=1. Otherwise, true_predictions[i]=0
    """
    n_pred = bboxes_pred.shape[0]
 # 예측 box가 TP이면 1, FP이면 0으로 기록
    true_predictions = th.zeros(n_pred, device=DEVICE)

    for pbox_ind, pbox in enumerate(bboxes_pred):
        n_gt = bboxes_gt.shape[0]
        pbox_coords = pbox[2:]
        bboxes_gt_coords = bboxes_gt[:, 1:]
        pbox_class_ind = pbox[0].long()
        bboxes_gt_class_ind = bboxes_gt[:, 0].long()

        iou_scores = th.zeros(n_gt, device=DEVICE)
        same_class = pbox_class_ind == bboxes_gt_class_ind
        iou_scores[same_class] = iou(pbox_coords, bboxes_gt_coords[same_class])

        if iou_scores.shape[0]:
 # 가장 많이 겹치는 ground truth box를 찾습니
            best_iou, gt_box_ind = th.max(iou_scores, dim=0)
            if best_iou > IOU_THRESHOLD:
                true_predictions[pbox_ind] = 1
                bboxes_gt = bboxes_gt[th.arange(n_gt, device=DEVICE) != gt_box_ind]

    return true_predictions


# test set 전체에 대해 class별 AP와 mAP를 계산하는 함수
def evaluate_model(model: YOLOv1, test_loader: DataLoader) -> Tuple[float, List[float]]:
    """
    To evaluate the performance of the trained YOLOv1 model on the test set, we calculate the mean average precision
    (map) metric. To calculate the map metric, the average precisions are first computed for each of the classes of the
    dataset. For each class, the bounding boxes predictions are sorted in descending order with respect to their
    confidence values. In this way, the class precision and recall are computed. The class precision is interpolated,
    such that the precision p(r) at recall r: p(r) = max(p(r')), where r' >= r.

    :param model: The trained YOLOv1 (detection) model
    :param test_loader: The DataLoader of the PASCAL VOC test set
    :return: The mean average precision of the model and the average precisions for each of the classes of the PASCAL
             VOC dataset.
    """
    total_class_pred_bboxes = th.zeros(VOC_Detection.C, device=DEVICE)
    total_class_gt_bboxes = th.zeros(VOC_Detection.C, device=DEVICE)
    total_predictions = th.empty((0, 3), device=DEVICE)

 # 시각화/평가 단계에서는 gradient 계산이 필요 없으므로 비활성화
    with th.no_grad():
 # BatchNorm/Dropout 등을 평가 모드로 전환
    model.eval()
        for x, bboxes_gt in test_loader:
            x, bboxes_gt = x.to(DEVICE), bboxes_gt.to(DEVICE).squeeze(0)
            y = model(x)
            bboxes_pred = postprocessing(y,
                                         prob_threshold=PROB_THRESHOLD,
                                         conf_mode=CONF_MODE,
                                         nms_threshold=NMS_THESHOLD)

            total_class_pred_bboxes += th.bincount(bboxes_pred[:, 0].long(), minlength=VOC_Detection.C)
            total_class_gt_bboxes += th.bincount(bboxes_gt[:, 0].long(), minlength=VOC_Detection.C)

            predictions_class_ind = bboxes_pred[:, 0]
            predictions_conf = bboxes_pred[:, 1]
            is_true_pred_bbox = evaluate_predictions(bboxes_gt, bboxes_pred)

            sample_predictions = th.stack([predictions_class_ind, predictions_conf, is_true_pred_bbox], dim=-1)
            total_predictions = th.cat([total_predictions, sample_predictions])

    average_precisions = []
    for c in range(VOC_Detection.C):

        class_mask = total_predictions[:, 0] == c
        if not th.max(class_mask):
            continue

        class_predictions = total_predictions[class_mask]
        sort_ind = th.argsort(class_predictions[:, 1], descending=True)
        sorted_tp = class_predictions[sort_ind, 2]
        cumsum_tp = th.cumsum(sorted_tp, dim=0)

        class_precision = cumsum_tp / th.arange(start=1, end=total_class_pred_bboxes[c] + 1, device=DEVICE)
        class_precision = th.flip(th.cummax(th.flip(class_precision, [0]), dim=0)[0], [0])

        class_recall = cumsum_tp / total_class_gt_bboxes[c]
        class_recall = th.cat([th.zeros(1, device=DEVICE), class_recall], dim=0)

        class_ap = th.sum(class_precision * (class_recall[1:] - class_recall[:-1]))
        average_precisions.append(class_ap.item() * 100)

    mAP = sum(average_precisions) / len(average_precisions)
    return mAP, average_precisions


# 모델과 테스트 데이터셋을 준비하는 함수
def setup_evaluation() -> Tuple[YOLOv1, DataLoader]:
    """
    Instantiate the model, the PASCAL VOC test dataset and the corresponding loader. The model's weights are loaded from
    the checkpoint file that was updated at the end of the training.

    :return: The YOLOv1 (detection) model and the DataLoader of the PASCAL VOC test set.
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
    Plot a horizontal bar plot, where each bar corresponds to a class of the VOC dataset and has width equal to the
    class' average precision.

    :param average_precisions: A list that contains the average precisions for each class of the VOC detection dataset.
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
    model, test_loader = setup_evaluation()
    mAP, average_precisions = evaluate_model(model, test_loader)
    print(f'Mean Average Precision = {mAP:.1f}%')

    if PLOT:
        plot_class_ap(average_precisions)


if __name__ == '__main__':
    main()