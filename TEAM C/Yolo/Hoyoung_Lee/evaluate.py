def non_max_suppression(boxes: th.Tensor, nms_threshold: float) -> th.Tensor:
    """
    NMS (Non-Maximum Suppression): 
    하나의 객체를 여러 그리드 셀이 동시에 탐지했을 때 발생하는 중복 박스를 제거하는 함수
    """
    nms_boxes = []
    # 1. 예측된 박스들을 신뢰도(Confidence) 순으로 내림차순 정렬
    sort_ind = th.argsort(boxes[:, 1], descending=True)
    boxes = boxes[sort_ind, :]
    
    while len(boxes):
        # 2. 가장 신뢰도가 높은 박스는 결과 리스트에 무조건 저장
        box1, boxes = boxes[0], boxes[1:]
        nms_boxes.append(box1)

        # 3. 방금 저장한 박스와 나머지 박스들의 IOU(겹치는 비율)를 계산하여
        # 임계값(nms_threshold) 이상 겹치면서 같은 클래스를 예측한 박스들은 제거 (중복 탐지로 간주)
        valid_boxes = iou_scores < nms_threshold
        boxes = boxes[valid_boxes]

    return boxes

def evaluate_model(model: YOLOv1, test_loader: DataLoader) -> Tuple[float, List[float]]:
    # mAP (Mean Average Precision) 계산 로직
    # 각 클래스별로 정밀도(Precision)와 재현율(Recall)을 구해 모델 성능 평가
    # ...