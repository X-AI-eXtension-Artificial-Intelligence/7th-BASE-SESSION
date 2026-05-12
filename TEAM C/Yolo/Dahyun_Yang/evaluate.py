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
S = 7 # 이미지를 7x7 grid로 나눔
B = 2 # 이미지를 7x7 grid로 나눔
D = 448 # 입력 이미지 크기 448x448

# Data Loading Hyperparameters
MINI_BATCH = 1
NUM_WORKERS = 1
PIN_MEMORY = True

# VOC Dataset Directory
PASCAL_VOC_DIR_PATH = "/media/soul/DATA/cv_datasets/PASCAL_VOC/VOC_Detection"

# Trained Model Path
TRAINED_MODEL_WEIGHTS = r"C:\Users\2023user\Downloads\yolov1_pytorch-main\yolov1_pytorch-main\checkpoints\trained_model_weights.pt"

# Compute Device (use a GPU if available)
DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

# Postprocessing Hyperparameters(평가 후처리 사용되는 값)
PROB_THRESHOLD = 0.005  
# this value is set ~= 0 for the map metric calculation
# confidence가 이 값보다 낮으면 box 제거
# = mAP 계산에서는 다양한 confidence 수준의 prediction을 최대한 포함해야 하기 때문에 threshold를 낮게 둔 것
NMS_THESHOLD = 0.6 # NMS에서 겹치는 box 제거 기준
IOU_THRESHOLD = 0.5 #  예측 box가 정답 box와 match되었다고 보는 기준
CONF_MODE = 'class' # confidence 계산 방식

# Plot the classes' APs
PLOT = True

# YOLO 모델 출력값에서 실제 detection box 후보를 뽑아내는 함수
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

    assert conf_mode in ['objectness', 'class']
    # class score에 softmax를 적용
    y[..., :VOC_Detection.C] = F.softmax(y[..., :VOC_Detection.C], dim=-1)
    # 각 grid cell에서 가장 확률이 높은 class
    class_score, class_ind = th.max(y[..., :VOC_Detection.C], dim=-1)
    # 각 grid cell에 있는 B=2개의 box 중 confidence/objectness가 가장 높은 box 하나만 선택
    objectness, bboxes_ind = th.max(y[..., [VOC_Detection.C + i * 5 for i in range(B)]], dim=-1)
    bboxes_coords_ind = th.arange(4, device=DEVICE)[None, None, None, :] + VOC_Detection.C + bboxes_ind[
        ..., None] * 5 + 1
    bboxes_coords = th.gather(y, dim=-1, index=bboxes_coords_ind)
    # YOLOv1은 한 grid cell마다 box를 2개 예측하지만, 이 코드에서는 최종적으로 각 grid cell당 가장 좋은 box 하나만 사용
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

# 같은 object를 여러 box가 중복 검출했을 때 하나만 남기는 과정
def non_max_suppression(boxes: th.Tensor,
                        nms_threshold: float) -> th.Tensor:
    """
    1. confidence가 높은 순서대로 box 정렬
    2. 가장 높은 confidence box를 선택
    3. 같은 class이면서 IoU가 높은 box 제거
    4. 남은 box들에 대해 반복
    """
    nms_boxes = []
    # confidence 기준으로 내림차순 정렬
    sort_ind = th.argsort(boxes[:, 1], descending=True)
    boxes = boxes[sort_ind, :]
    while len(boxes):
        box1, boxes = boxes[0], boxes[1:]
        nms_boxes.append(box1)

        box1_class, box1_coords = box1[0], box1[2:]
        iou_scores = th.zeros(len(boxes), device=DEVICE)
        # 같은 class인 box끼리만 IoU를 계산
        same_class = box1_class == boxes[:, 0]
        iou_scores[same_class] = iou(box1_coords, boxes[same_class][:, 2:])
        # IoU가 threshold 이상이면 같은 물체를 잡은 중복 box라고 보고 제거
        valid_boxes = iou_scores < nms_threshold
        boxes = boxes[valid_boxes]

    if len(nms_boxes):
        boxes = th.stack(nms_boxes, dim=0)
    else:
        boxes = th.empty((0, 6), device=DEVICE)
    return boxes

# YOLO 출력 bbox 좌표를 실제 이미지 좌표계로 변환
def rescale_bboxes(y: th.Tensor) -> None:
    """
    The coordinates of the predicted bounding boxes are transformed. Each of the predicted bounding boxes of the YOLO
    model are represented as (x_center_norm, y_center_norm, sqrt(width_norm), sqrt(height_norm)). The predicted center
    coordinates x_center_norm and y_center_norm are normalized by the dimension of the grid cells, while the width_norm
    and height_norm are normalized by the dimension of the resized image. The function rescales these coordinates to
    the (x_center, y_center, width, height) by reversing the normalizations.
    -> x, y는 해당 grid cell 안에서의 상대 좌표이고, w, h는 이미지 전체 크기에 대해 normalize된 값

    :param y: The predictions of the YOLOv1 model
    """
    row, col = th.meshgrid(th.arange(S, device=DEVICE), th.arange(S, device=DEVICE), indexing='ij')
    row = row.unsqueeze(-1)
    col = col.unsqueeze(-1)
 
    # 이를 다시 실제 이미지 크기 기준 좌표로 되돌림
    y[..., [VOC_Detection.C + i * 5 + 1 for i in range(B)]] += col
    y[..., [VOC_Detection.C + i * 5 + 2 for i in range(B)]] += row
    # grid cell 내부 좌표에 cell 위치를 더함
    # grid 단위 좌표를 pixel 단위 좌표로 변환
    y[..., [VOC_Detection.C + i * 5 + j for j in [1, 2] for i in range(B)]] *= D / S
    # width, height는 제곱해서 원래 크기로 되돌림
    y[..., [VOC_Detection.C + i * 5 + j for j in [3, 4] for i in range(B)]] *= D * y[..., [VOC_Detection.C + i * 5 + j
                                                                                           for j in [3, 4] for i in
                                                                                           range(B)]]

# 모델 출력값을 최종 예측 box로 바꾸는 전체 후처리 함수
def postprocessing(y: th.Tensor,
                   prob_threshold: float,
                   conf_mode: Literal['objectness', 'class'],
                   nms_threshold: float) -> th.Tensor:
    """
    1. YOLO 좌표를 실제 이미지 좌표로 변환
    2. confidence 낮은 box 제거
    3. NMS로 중복 box 제거
    """
    rescale_bboxes(y)
    boxes = get_detected_boxes(y, prob_threshold, conf_mode)
    boxes = non_max_suppression(boxes, nms_threshold)
    return boxes
    # 최종 출력: [class_index, confidence, xmin, ymin, xmax, ymax]

# 한 이미지에 대해 예측 box가 정답인지 아닌지 판단
def evaluate_predictions(bboxes_gt: th.Tensor, bboxes_pred: th.Tensor) -> th.Tensor:
    """
    1. 예측 class와 ground truth class가 같아야 함
    2. 예측 box와 ground truth box의 IoU가 IOU_THRESHOLD보다 커야 함
    3. 하나의 ground truth box는 하나의 prediction에만 매칭됨
    """
    n_pred = bboxes_pred.shape[0]
    true_predictions = th.zeros(n_pred, device=DEVICE)

    for pbox_ind, pbox in enumerate(bboxes_pred):
        n_gt = bboxes_gt.shape[0]
        pbox_coords = pbox[2:]
        bboxes_gt_coords = bboxes_gt[:, 1:]
        pbox_class_ind = pbox[0].long()
        bboxes_gt_class_ind = bboxes_gt[:, 0].long()

        iou_scores = th.zeros(n_gt, device=DEVICE)
        # class가 같은 ground truth box에 대해서만 IoU를 계산
        same_class = pbox_class_ind == bboxes_gt_class_ind
        iou_scores[same_class] = iou(pbox_coords, bboxes_gt_coords[same_class])

        if iou_scores.shape[0]:
            best_iou, gt_box_ind = th.max(iou_scores, dim=0)
            if best_iou > IOU_THRESHOLD:
                # IoU가 기준보다 높으면 true positive로 처리, 매칭된 ground truth box는 제거
                true_predictions[pbox_ind] = 1
                bboxes_gt = bboxes_gt[th.arange(n_gt, device=DEVICE) != gt_box_ind]

    return true_predictions

# 전체 평가의 핵심
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

    with th.no_grad():
        model.eval()
        for x, bboxes_gt in test_loader:
            x, bboxes_gt = x.to(DEVICE), bboxes_gt.to(DEVICE).squeeze(0)
            # 모델 예측한 후, 후처리, ground truth와 비교, TP/FP 판단(모든 이미지의 prediction을 모아서 class 별 AP 계산)
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
        # class별 prediction을 confidence 높은 순서대로 정렬
        class_predictions = total_predictions[class_mask]
        sort_ind = th.argsort(class_predictions[:, 1], descending=True)
        sorted_tp = class_predictions[sort_ind, 2]
        # 누적 true positive를 계산
        cumsum_tp = th.cumsum(sorted_tp, dim=0)

        class_precision = cumsum_tp / th.arange(start=1, end=total_class_pred_bboxes[c] + 1, device=DEVICE)
        class_precision = th.flip(th.cummax(th.flip(class_precision, [0]), dim=0)[0], [0])

        class_recall = cumsum_tp / total_class_gt_bboxes[c]
        class_recall = th.cat([th.zeros(1, device=DEVICE), class_recall], dim=0)

        class_ap = th.sum(class_precision * (class_recall[1:] - class_recall[:-1]))
        average_precisions.append(class_ap.item() * 100)

    mAP = sum(average_precisions) / len(average_precisions)
    return mAP, average_precisions


def setup_evaluation() -> Tuple[YOLOv1, DataLoader]:
    # 평가에 필요한 모델과 test loader를 준비
    """
    Instantiate the model, the PASCAL VOC test dataset and the corresponding loader. The model's weights are loaded from
    the checkpoint file that was updated at the end of the training.

    :return: The YOLOv1 (detection) model and the DataLoader of the PASCAL VOC test set.
    """
    model = YOLOv1(S=S,
                   B=B,
                   C=VOC_Detection.C).to(DEVICE)
    trained_model_weights = th.load(TRAINED_MODEL_WEIGHTS, weights_only=False)
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
    # test image도 학습 때와 동일하게 448x448로 resize하고 tensor로 변환한다는 것이 중요함

# 시각화
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
