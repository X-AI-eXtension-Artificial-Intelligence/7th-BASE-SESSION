"""
inference.py  ─  YOLOv1 추론 스크립트
──────────────────────────────────────
사용법:
  python inference.py --image data/my_photo.jpg --weights weights/best.pt
  python inference.py --image data/my_photo.jpg  # 랜덤 가중치로 데모 추론
"""

import argparse
import os
import cv2
import numpy as np
import torch

from model import YOLOv1
from utils import decode_predictions, nms, draw_boxes, VOC_CLASSES

IMG_SIZE   = 448
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
CONF_THRESH = 0.25
IOU_THRESH  = 0.45


def parse_args():
    p = argparse.ArgumentParser(description='YOLOv1 추론')
    p.add_argument('--image',   type=str, required=True, help='입력 이미지 경로')
    p.add_argument('--weights', type=str, default=None,  help='체크포인트 경로')
    p.add_argument('--output',  type=str, default='outputs/result.jpg', help='결과 저장 경로')
    p.add_argument('--conf',    type=float, default=CONF_THRESH)
    p.add_argument('--iou',     type=float, default=IOU_THRESH)
    return p.parse_args()


def load_model(weights_path: str = None, S=7, B=2, C=20) -> torch.nn.Module:
    model = YOLOv1(S=S, B=B, C=C, mode='detection').to(DEVICE)

    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        print(f"[inference] 가중치 로드: {weights_path}  (epoch {ckpt.get('epoch', '?')})")
    else:
        print("[inference] 가중치 없음 → 랜덤 초기화 상태로 추론 (데모)")

    model.eval()
    return model


def preprocess(image_path: str):
    """이미지 → (1, 3, 448, 448) Tensor + 원본 이미지"""
    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"이미지를 열 수 없습니다: {image_path}"
    orig_h, orig_w = img_bgr.shape[:2]

    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor      = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    return tensor, img_bgr, orig_h, orig_w


def postprocess(output, orig_h, orig_w, conf_thresh, iou_thresh):
    """(1, S, S, C+B*5) → 스케일 복원된 최종 박스 리스트"""
    boxes_list = decode_predictions(output, conf_thresh=conf_thresh)
    raw_boxes  = boxes_list[0]                        # 첫 번째 이미지
    kept_boxes = nms(raw_boxes, iou_thresh=iou_thresh)

    # IMG_SIZE(448) → 원본 이미지 크기로 스케일 복원
    scale_x = orig_w / IMG_SIZE
    scale_y = orig_h / IMG_SIZE
    scaled = []
    for (x1, y1, x2, y2, score, cls_id) in kept_boxes:
        scaled.append((x1 * scale_x, y1 * scale_y,
                       x2 * scale_x, y2 * scale_y,
                       score, cls_id))
    return scaled


@torch.no_grad()
def run(args):
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # ── 모델 로드 ─────────────────────────────────────────────
    model = load_model(args.weights)

    # ── 전처리 ───────────────────────────────────────────────
    tensor, img_bgr, orig_h, orig_w = preprocess(args.image)

    # ── 추론 ─────────────────────────────────────────────────
    output = model(tensor)          # (1, 7, 7, C+B*5)
    print(f"[inference] 출력 shape: {output.shape}")

    # ── 후처리 ───────────────────────────────────────────────
    boxes = postprocess(output, orig_h, orig_w, args.conf, args.iou)
    print(f"[inference] 감지된 객체: {len(boxes)}개")
    for (x1, y1, x2, y2, score, cls_id) in boxes:
        print(f"  [{VOC_CLASSES[cls_id]}] score={score:.3f}  "
              f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    # ── 시각화 ───────────────────────────────────────────────
    result_img = draw_boxes(img_bgr, boxes)
    cv2.imwrite(args.output, result_img)
    print(f"[inference] 결과 저장 → {args.output}")


if __name__ == '__main__':
    args = parse_args()
    run(args)
