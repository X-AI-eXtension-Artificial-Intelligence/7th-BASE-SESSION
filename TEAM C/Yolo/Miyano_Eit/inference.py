import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from model import get_model


PASCAL_VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def decode_predictions(pred, S=7, B=2, C=20, conf_thresh=0.3):
    boxes = []

    for i in range(S):
        for j in range(S):
            for b in range(B):
                offset = b * 5
                x_cell = pred[i, j, offset].item()
                y_cell = pred[i, j, offset + 1].item()
                w = pred[i, j, offset + 2].item()
                h = pred[i, j, offset + 3].item()
                conf = pred[i, j, offset + 4].item()

                if conf < conf_thresh:
                    continue

                x_center = (j + x_cell) / S
                y_center = (i + y_cell) / S

                x1 = x_center - abs(w) / 2
                y1 = y_center - abs(h) / 2
                x2 = x_center + abs(w) / 2
                y2 = y_center + abs(h) / 2

                # 0~1 범위로 clamp
                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))

                # x1 < x2, y1 < y2 보장
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                class_probs = pred[i, j, B * 5:]
                class_id = class_probs.argmax().item()
                class_score = class_probs[class_id].item() * conf

                boxes.append((x1, y1, x2, y2, class_score,
                              PASCAL_VOC_CLASSES[class_id]))

    return boxes


def non_max_suppression(boxes, iou_thresh=0.5):
    """
    논문 Section 2.3: Non-maximal suppression으로 중복 박스 제거
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    kept = []

    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if compute_iou(best, b) < iou_thresh]

    return kept


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


def visualize(image_path, boxes, save_path="etc/result.jpg"):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    W, H = image.size

    for (x1, y1, x2, y2, conf, class_name) in boxes:
        # 픽셀 좌표로 변환
        px1, py1 = int(x1 * W), int(y1 * H)
        px2, py2 = int(x2 * W), int(y2 * H)

        # 좌표 정렬 (x1 < x2, y1 < y2 보장)
        px1, px2 = min(px1, px2), max(px1, px2)
        py1, py2 = min(py1, py2), max(py1, py2)

        # 너무 작은 박스 제외
        if px2 - px1 < 2 or py2 - py1 < 2:
            continue

        draw.rectangle([px1, py1, px2, py2], outline='red', width=3)
        draw.text((px1, max(0, py1 - 15)),
                  f"{class_name} {conf:.2f}",
                  fill='red')

    image.save(save_path)
    print(f"결과 저장 완료 → etc/result.jpg")
    return image

def detect(image_path, model_path="etc/yolo.pt",
           S=7, B=2, C=20, conf_thresh=0.3, iou_thresh=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = get_model(S=S, B=B, C=C).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)[0]                      # (S, S, B*5+C)

    # 예측 디코딩
    boxes = decode_predictions(pred, S, B, C, conf_thresh)

    # NMS 적용
    boxes = non_max_suppression(boxes, iou_thresh)

    print(f"\n탐지된 객체 수: {len(boxes)}")
    for box in boxes:
        print(f"  {box[5]:15s} | confidence: {box[4]:.3f} | "
              f"box: ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f})")

    # 시각화
    visualize(image_path, boxes)

    return boxes


if __name__ == "__main__":
    detect("dataset/images/test.jpg")