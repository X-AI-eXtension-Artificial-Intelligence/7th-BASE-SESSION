"""
inspect.py
학습된 모델의 Attention Rollout을 실제 이미지 위에 시각화한다.

사용법:
    python inspect.py
    python inspect.py --epoch 15   # 저장된 체크포인트 지정
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision

from config import CFG
from data   import get_transforms, CIFAR10_CLASSES
from vit    import build_vit
from utils  import attention_rollout


@torch.no_grad()
def visualize(model, img_tensor, label_idx, cfg):
    device     = cfg["device"]
    n_patches  = cfg["n_patches"]
    patch_side = int(n_patches ** 0.5)   # 8
    patch_size = cfg["patch_size"]        # 4
    img_size   = cfg["img_size"]          # 32

    model.eval()
    inp = img_tensor.unsqueeze(0).to(device)

    logits, attn_list = model(inp, return_attn=True)
    pred_idx = logits.argmax(1).item()

    # Attention Rollout → (N,) 정규화
    rollout  = attention_rollout(attn_list, cfg["seq_len"], device)
    cls_attn = rollout[0, 0, 1:].reshape(patch_side, patch_side).cpu().numpy()
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

    # 원본 이미지 복원 (정규화 역변환)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])
    raw  = img_tensor.permute(1, 2, 0).numpy()
    raw  = (raw * std + mean).clip(0, 1)

    # 어텐션 맵을 이미지 해상도로 업샘플
    attn_map = np.kron(cls_attn, np.ones((patch_size, patch_size)))  # (32, 32)

    # ── 플롯 ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"Label: {CIFAR10_CLASSES[label_idx]}  |  "
        f"Pred: {CIFAR10_CLASSES[pred_idx]}  |  "
        f"{'O' if pred_idx == label_idx else 'X'}",
        fontsize=13, fontweight="bold"
    )

    # 1) 원본 이미지
    axes[0].imshow(raw)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 2) 어텐션 히트맵
    im = axes[1].imshow(cls_attn, cmap="hot", interpolation="nearest",
                        vmin=0, vmax=1)
    axes[1].set_title("Attention Rollout (8×8 patches)")
    axes[1].set_xticks(range(patch_side))
    axes[1].set_yticks(range(patch_side))
    axes[1].set_xticklabels(range(patch_side), fontsize=7)
    axes[1].set_yticklabels(range(patch_side), fontsize=7)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 가장 높은 어텐션 패치에 빨간 테두리
    top_idx   = cls_attn.flatten().argmax()
    top_r, top_c = divmod(int(top_idx), patch_side)
    rect = patches.Rectangle(
        (top_c - 0.5, top_r - 0.5), 1, 1,
        linewidth=2, edgecolor="red", facecolor="none"
    )
    axes[1].add_patch(rect)

    # 3) 오버레이 (원본 + 어텐션)
    axes[2].imshow(raw)
    axes[2].imshow(attn_map, cmap="hot", alpha=0.5, interpolation="bilinear",
                   vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = "attention_result.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n저장 완료: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  type=str, default=CFG["save_path"],
                        help="모델 체크포인트 경로")
    parser.add_argument("--idx",   type=int, default=0,
                        help="테스트셋에서 볼 이미지 인덱스")
    parser.add_argument("--class_name", type=str, default=None,
                        help="특정 클래스만 찾아서 시각화 (예: cat)")
    args = parser.parse_args()

    device = CFG["device"]

    # 모델 로드
    print(f"체크포인트 로드 중: {args.ckpt}")
    model = build_vit(CFG).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    print("로드 완료.")

    # 테스트셋
    _, test_tf = get_transforms(CFG["img_size"])
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=test_tf
    )

    # 이미지 선택
    if args.class_name:
        target_cls = CIFAR10_CLASSES.index(args.class_name)
        indices = [i for i, (_, l) in enumerate(test_set) if l == target_cls]
        idx = indices[0] if indices else 0
        print(f"'{args.class_name}' 클래스 첫 번째 이미지 (index={idx}) 사용")
    else:
        idx = args.idx

    img_tensor, label = test_set[idx]
    print(f"이미지 index={idx}  /  정답: {CIFAR10_CLASSES[label]}")

    visualize(model, img_tensor, label, CFG)


if __name__ == "__main__":
    main()
