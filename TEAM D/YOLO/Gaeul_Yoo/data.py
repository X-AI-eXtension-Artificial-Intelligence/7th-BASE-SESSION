import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.datasets as dsets
from PIL import Image, ImageEnhance

from util import VOC_CLASSES, encode_targets

IMG_SIZE = 448


class VOCDataset(Dataset):
    """
    Wraps torchvision.datasets.VOCDetection so that each sample returns
    (image_tensor, target_tensor) ready for YOLOLoss.

    image_tensor : (3, 448, 448)  ImageNet-normalized
    target_tensor: (S, S, B*5+C)  from encode_targets()
    """

    def __init__(self, root, year='2007', image_set='train',
                 augment=True, S=7, B=2, C=20):
        self.augment = augment
        self.S, self.B, self.C = S, B, C

        self.voc = dsets.VOCDetection(
            root=root, year=year, image_set=image_set,
            download=True,
        )
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, annotation = self.voc[idx]
        img = img.convert('RGB')

        # ── Parse XML annotation ───────────────────────────────────────────
        ann    = annotation['annotation']
        img_w  = int(ann['size']['width'])
        img_h  = int(ann['size']['height'])

        objects = ann['object']
        if isinstance(objects, dict):       # single object → wrap in list
            objects = [objects]

        boxes, labels = [], []
        for obj in objects:
            name = obj['name']
            if name not in VOC_CLASSES:
                continue
            # skip difficult instances (optional)
            if int(obj.get('difficult', 0)):
                continue

            bnd  = obj['bndbox']
            x1   = float(bnd['xmin']) / img_w
            y1   = float(bnd['ymin']) / img_h
            x2   = float(bnd['xmax']) / img_w
            y2   = float(bnd['ymax']) / img_h
            # clamp to [0,1] in case of annotation noise
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(1.0, x2), min(1.0, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(VOC_CLASSES.index(name))

        # ── Augmentation ──────────────────────────────────────────────────
        if self.augment and boxes:
            img, boxes, labels = _augment(img, boxes, labels)

        # ── Resize → tensor ───────────────────────────────────────────────
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_tensor = self.to_tensor(img)

        target = encode_targets(boxes, labels, self.S, self.B, self.C)
        return img_tensor, target


# ─── Augmentation (paper §2.4) ────────────────────────────────────────────────

def _augment(img, boxes, labels):
    """
    Applies the augmentation described in YOLO §2.4:
      - Random scaling up to ±20 %
      - Random translation up to ±20 %
      - Random horizontal flip  (not in paper; standard practice)
      - Random exposure/saturation jitter up to ×1.5
    """
    w, h = img.size

    # ── Horizontal flip ───────────────────────────────────────────────────
    if random.random() < 0.5:
        img    = img.transpose(Image.FLIP_LEFT_RIGHT)
        boxes  = [[1.0 - b[2], b[1], 1.0 - b[0], b[3]] for b in boxes]

    # ── Scale + translate ─────────────────────────────────────────────────
    scale = random.uniform(0.8, 1.2)
    dx    = random.uniform(-0.2, 0.2)   # fraction of original width
    dy    = random.uniform(-0.2, 0.2)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img   = img.resize((new_w, new_h), Image.BILINEAR)

    # Paste scaled image onto a grey canvas of original size
    canvas = Image.new('RGB', (w, h), (128, 128, 128))
    off_x  = int(dx * w)
    off_y  = int(dy * h)
    canvas.paste(img, (off_x, off_y))

    new_boxes, new_labels = [], []
    for b, l in zip(boxes, labels):
        x1 = (b[0] * new_w + off_x) / w
        y1 = (b[1] * new_h + off_y) / h
        x2 = (b[2] * new_w + off_x) / w
        y2 = (b[3] * new_h + off_y) / h
        x1, x2 = max(0.0, x1), min(1.0, x2)
        y1, y2 = max(0.0, y1), min(1.0, y2)
        if x2 > x1 and y2 > y1:
            new_boxes.append([x1, y1, x2, y2])
            new_labels.append(l)

    # ── Exposure / saturation jitter ──────────────────────────────────────
    canvas = ImageEnhance.Color(canvas).enhance(random.uniform(0.5, 1.5))
    canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.5, 1.5))

    return canvas, new_boxes, new_labels
