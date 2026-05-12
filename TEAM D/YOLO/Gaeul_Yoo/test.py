import os
import argparse
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from PIL import Image

from model import YOLOv1
from data import VOCDataset, IMG_SIZE
from util import VOC_CLASSES, decode_predictions, iou_tensor

S, B, C = 7, 2, 20

# One color per class for visualization
COLORS = plt.cm.get_cmap('tab20', C)


# ─── mAP ──────────────────────────────────────────────────────────────────────

def compute_map(model, loader, device, iou_thresh=0.5, conf_thresh=0.25):
    """
    Computes VOC-style mAP@iou_thresh over the given DataLoader.

    Returns the mAP value (float).
    """
    model.eval()

    # per-class: list of (confidence, is_tp)
    detections = {c: [] for c in range(C)}
    gt_counts  = {c: 0  for c in range(C)}

    with torch.no_grad():
        for imgs, targets in loader:
            imgs   = imgs.to(device)
            preds  = model(imgs).cpu()

            for i in range(imgs.size(0)):
                det_list = decode_predictions(
                    preds[i], S=S, B=B, C=C,
                    conf_thresh=conf_thresh
                )

                # ── Recover GT boxes from the target tensor ────────────────
                gt_by_cls = {c: [] for c in range(C)}
                tgt = targets[i]    # (S, S, B*5+C)
                for row in range(S):
                    for col in range(S):
                        if tgt[row, col, 4] < 0.5:
                            continue
                        tx = tgt[row, col, 0].item()
                        ty = tgt[row, col, 1].item()
                        w  = tgt[row, col, 2].item()
                        h  = tgt[row, col, 3].item()
                        cx = (col + tx) / S
                        cy = (row + ty) / S
                        x1, y1 = cx - w / 2, cy - h / 2
                        x2, y2 = cx + w / 2, cy + h / 2
                        cls = int(tgt[row, col, B * 5:].argmax())
                        gt_by_cls[cls].append([x1, y1, x2, y2])
                        gt_counts[cls] += 1

                # ── Match each detection to GT (greedy, by confidence) ─────
                matched = {c: set() for c in range(C)}
                for det in sorted(det_list, key=lambda x: -x[4]):
                    x1, y1, x2, y2, score, cls = det
                    cls = int(cls)
                    gts = gt_by_cls.get(cls, [])

                    best_iou, best_j = 0.0, -1
                    if gts:
                        iou_mat = iou_tensor(
                            torch.tensor([[x1, y1, x2, y2]]),
                            torch.tensor(gts, dtype=torch.float32),
                        )  # (1, len(gts))
                        best_iou = iou_mat[0].max().item()
                        best_j   = iou_mat[0].argmax().item()

                    is_tp = int(
                        best_iou >= iou_thresh and best_j not in matched[cls]
                    )
                    if is_tp:
                        matched[cls].add(best_j)
                    detections[cls].append((score, is_tp))

    # ── Per-class AP ───────────────────────────────────────────────────────
    aps = []
    print(f'\n{"Class":<16} AP')
    print('-' * 26)
    for c in range(C):
        ngt = gt_counts[c]
        if ngt == 0:
            continue
        dets = sorted(detections[c], key=lambda x: -x[0])
        tp   = np.array([d[1] for d in dets], dtype=float)
        fp   = 1.0 - tp
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        rec  = tp_c / (ngt + 1e-6)
        pre  = tp_c / (tp_c + fp_c + 1e-6)
        ap   = _voc_ap(rec, pre)
        aps.append(ap)
        print(f'{VOC_CLASSES[c]:<16} {ap:.3f}')

    mAP = float(np.mean(aps)) if aps else 0.0
    print('-' * 26)
    print(f'mAP@{iou_thresh:<4}         {mAP:.4f}\n')
    return mAP


def _voc_ap(recall, precision):
    """11-point VOC interpolation."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t].max() if (recall >= t).any() else 0.0
        ap += p / 11
    return ap


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize(model, dataset, device, n=8, conf_thresh=0.25, save_dir=None):
    """Draw predicted bounding boxes on n random images from dataset."""
    model.eval()
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for idx in indices:
        img_tensor, _ = dataset[idx]

        with torch.no_grad():
            pred = model(img_tensor.unsqueeze(0).to(device)).cpu()[0]

        results = decode_predictions(pred, S=S, B=B, C=C,
                                     conf_thresh=conf_thresh)

        # Undo normalization for display
        img_show = (img_tensor * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()

        fig, ax = plt.subplots(1, figsize=(7, 7))
        ax.imshow(img_show)
        ax.axis('off')

        for det in results:
            x1, y1, x2, y2, score, cls = det
            cls   = int(cls)
            color = COLORS(cls)
            rect  = patches.Rectangle(
                (x1 * IMG_SIZE, y1 * IMG_SIZE),
                (x2 - x1) * IMG_SIZE, (y2 - y1) * IMG_SIZE,
                linewidth=2, edgecolor=color, facecolor='none',
            )
            ax.add_patch(rect)
            ax.text(
                x1 * IMG_SIZE, max(0, y1 * IMG_SIZE - 4),
                f'{VOC_CLASSES[cls]}  {score:.2f}',
                color='white', fontsize=8,
                bbox=dict(facecolor=color, alpha=0.7, pad=2, linewidth=0),
            )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f'vis_{idx:05d}.png'),
                        bbox_inches='tight', dpi=150)
        plt.tight_layout()
        plt.show()
        plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv1 on Pascal VOC')
    parser.add_argument('--weights',    required=True,
                        help='path to .pth checkpoint')
    parser.add_argument('--data_root',  default='./data')
    parser.add_argument('--year',       default='2007')
    parser.add_argument('--split',      default='test',
                        help='trainval / test')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--iou_thresh', type=float, default=0.5)
    parser.add_argument('--conf_thresh',type=float, default=0.25)
    parser.add_argument('--visualize',  action='store_true')
    parser.add_argument('--vis_n',      type=int, default=8)
    parser.add_argument('--vis_dir',    default='./vis')
    args = parser.parse_args()

    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_set = VOCDataset(args.data_root, year=args.year,
                          image_set=args.split, augment=False)
    loader   = DataLoader(test_set, batch_size=args.batch_size,
                          shuffle=False, num_workers=4)

    model = YOLOv1(S=S, B=B, C=C).to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    # support both raw state-dict and {'model': ...} format
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    print(f'Loaded weights from {args.weights}')

    compute_map(model, loader, device,
                iou_thresh=args.iou_thresh,
                conf_thresh=args.conf_thresh)

    if args.visualize:
        visualize(model, test_set, device,
                  n=args.vis_n,
                  conf_thresh=args.conf_thresh,
                  save_dir=args.vis_dir)


if __name__ == '__main__':
    main()
