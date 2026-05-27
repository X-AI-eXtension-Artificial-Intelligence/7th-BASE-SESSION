"""
utils.py
학습 유틸리티 모음.

  get_scheduler       : 코사인 + 선형 웜업 스케줄러
  count_params        : 파라미터 수 계산
  print_model_info    : 모델 구조 요약 출력
  attention_rollout   : Attention Rollout 계산 (Abnar & Zuidema, 2020)
  show_attention      : 배치 한 장에 대한 어텐션 히트맵 출력
"""

import math
import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# 학습률 스케줄러: 선형 웜업 + 코사인 감쇠
# ──────────────────────────────────────────────
def get_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
):
    """
    step 기반 LambdaLR.
    - 0 ~ warmup_steps  : 0 → 1 선형 증가
    - warmup_steps ~ end: 코사인 감쇠 (1 → ~0)
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────
# 파라미터 수 계산
# ──────────────────────────────────────────────
def count_params(model: nn.Module):
    """전체 / 학습 가능 파라미터 수를 반환한다."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ──────────────────────────────────────────────
# 모델 정보 출력
# ──────────────────────────────────────────────
def print_model_info(model: nn.Module, cfg: dict):
    total, trainable = count_params(model)
    sep = "=" * 55
    print(sep)
    print("  Vision Transformer (ViT) — CIFAR-10 from Scratch")
    print(sep)
    print(f"  이미지 크기     : {cfg['img_size']}×{cfg['img_size']}")
    print(f"  패치 크기       : {cfg['patch_size']}×{cfg['patch_size']}")
    print(f"  패치 수  (N)    : {cfg['n_patches']}")
    print(f"  시퀀스 길이     : {cfg['seq_len']}  (N + [CLS])")
    print(f"  임베딩 차원 D   : {cfg['d_model']}")
    print(f"  Transformer 깊이 : {cfg['depth']} 블록")
    print(f"  어텐션 헤드 수  : {cfg['n_heads']}")
    print(f"  MLP ratio       : {cfg['mlp_ratio']}")
    print(f"  전체 파라미터   : {total:,}")
    print(f"  학습 파라미터   : {trainable:,}")
    print(f"  디바이스        : {cfg['device']}")
    print(sep)


# ──────────────────────────────────────────────
# Attention Rollout
#   Abnar & Zuidema (ACL 2020)
#   모든 레이어의 어텐션을 재귀적으로 곱해
#   최종 [CLS] 토큰이 각 패치에 갖는 어텐션을 구한다.
#
#   A'_ℓ = 0.5 × A_ℓ + 0.5 × I  (잔차 연결 반영)
#   R = A'_L × A'_{L-1} × ... × A'_1
# ──────────────────────────────────────────────
def attention_rollout(attn_list: list, seq_len: int, device) -> torch.Tensor:
    """
    Args:
        attn_list : list of (B, n_heads, N+1, N+1) — 각 레이어 어텐션 가중치
        seq_len   : N+1 (패치 수 + CLS)
        device    : torch device

    Returns:
        rollout : (B, N+1, N+1) — 레이어 전체 누적 어텐션
    """
    B = attn_list[0].size(0)
    rollout = torch.eye(seq_len, device=device).unsqueeze(0).expand(B, -1, -1)

    for attn in attn_list:
        # 헤드 평균: (B, N+1, N+1)
        attn_mean = attn.mean(dim=1)

        # 잔차 연결 반영: A' = 0.5A + 0.5I
        eye = torch.eye(seq_len, device=device).unsqueeze(0)
        attn_mean = 0.5 * attn_mean + 0.5 * eye

        # 행 정규화
        attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)

        rollout = attn_mean @ rollout

    return rollout  # (B, N+1, N+1)


# ──────────────────────────────────────────────
# Attention 히트맵 출력
# ──────────────────────────────────────────────
@torch.no_grad()
def show_attention(model, loader, cfg: dict):
    """
    테스트 배치 첫 번째 이미지에 대해
    Attention Rollout 결과를 콘솔에 출력한다.
    """
    device    = cfg["device"]
    seq_len   = cfg["seq_len"]
    n_patches = cfg["n_patches"]
    patch_side = int(n_patches ** 0.5)

    model.eval()
    imgs, labels = next(iter(loader))
    img = imgs[:1].to(device)

    _, attn_list = model(img, return_attn=True)

    rollout  = attention_rollout(attn_list, seq_len, device)
    cls_attn = rollout[0, 0, 1:]                # [CLS] → 패치들 (N,)
    cls_attn = cls_attn.reshape(patch_side, patch_side).cpu()

    # 정규화
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

    from data import CIFAR10_CLASSES
    print(f"\n[Attention Rollout]  정답 클래스: {CIFAR10_CLASSES[labels[0].item()]}")
    print(f"패치 그리드: {patch_side}×{patch_side}  (값 높을수록 집중)")
    print("-" * (patch_side * 7))

    for row in cls_attn.numpy():
        print("  " + "  ".join(f"{v:.3f}" for v in row))

    top_idx = cls_attn.flatten().argmax().item()
    top_r, top_c = divmod(top_idx, patch_side)
    print(f"\n  가장 높은 어텐션 패치: row={top_r}, col={top_c}  (index={top_idx})")
