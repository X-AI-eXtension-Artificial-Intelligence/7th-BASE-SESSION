# =============================================================================
# util/attention_visualizer.py
# Attention 가중치 시각화 유틸리티
# 추가 기능: 논문 공부 시 각 헤드가 어떤 관계를 학습했는지 확인 가능
# =============================================================================

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os


def visualize_attention(attention, src_tokens, trg_tokens,
                        layer=0, save_dir="saved/attention_maps"):
    """
    Multi-Head Attention 가중치를 시각화합니다.

    논문에서 설명한 것처럼:
    - 각 헤드가 서로 다른 관계를 학습하는지 확인
    - 헤드별로 어떤 토큰에 집중하는지 시각화

    Args:
        attention  : [n_heads, src_len, trg_len] Attention 가중치 행렬
        src_tokens : 소스 문장 토큰 리스트 (예: ["I", "like", "apples"])
        trg_tokens : 타겟 문장 토큰 리스트 (예: ["나는", "사과를", "좋아한다"])
        layer      : 시각화할 레이어 번호
        save_dir   : 이미지 저장 경로
    """

    os.makedirs(save_dir, exist_ok=True)

    n_heads = attention.size(0)

    # 헤드 수에 따라 subplot 크기 조절
    cols = 4
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle(f'Layer {layer} - Multi-Head Attention 시각화\n'
                 f'(각 헤드가 서로 다른 관계를 학습함을 확인)',
                 fontsize=14, y=1.02)

    axes = axes.flatten() if n_heads > 1 else [axes]

    for head in range(n_heads):
        ax = axes[head]

        # attention 가중치 추출 [src_len, trg_len]
        attn_weights = attention[head].detach().cpu().numpy()

        # 히트맵 그리기
        im = ax.matshow(attn_weights, cmap='Blues', vmin=0, vmax=1)

        # 축 레이블 설정
        ax.set_xticks(range(len(trg_tokens)))
        ax.set_yticks(range(len(src_tokens)))
        ax.set_xticklabels(trg_tokens, rotation=45, ha='left', fontsize=8)
        ax.set_yticklabels(src_tokens, fontsize=8)

        ax.set_xlabel('Target', fontsize=9)
        ax.set_ylabel('Source', fontsize=9)
        ax.set_title(f'Head {head + 1}', fontsize=10)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 사용하지 않는 subplot 숨기기
    for idx in range(n_heads, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    save_path = os.path.join(save_dir, f'layer{layer}_attention.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"  → Attention 시각화 저장: {save_path}")
    return save_path


def visualize_all_layers(attentions, src_tokens, trg_tokens,
                         save_dir="saved/attention_maps"):
    """
    모든 레이어의 Attention을 한 번에 시각화합니다.

    Args:
        attentions : list of [n_heads, src_len, trg_len]  (레이어별)
        src_tokens : 소스 문장 토큰 리스트
        trg_tokens : 타겟 문장 토큰 리스트
        save_dir   : 이미지 저장 경로
    """
    print(f"\n[Attention 시각화]")
    print(f"  소스 문장: {' '.join(src_tokens)}")
    print(f"  타겟 문장: {' '.join(trg_tokens)}")
    print(f"  총 {len(attentions)}개 레이어 시각화 중...")

    saved_paths = []
    for layer_idx, attention in enumerate(attentions):
        path = visualize_attention(
            attention, src_tokens, trg_tokens,
            layer=layer_idx + 1,
            save_dir=save_dir
        )
        saved_paths.append(path)

    print(f"  완료! 저장 위치: {save_dir}/")
    return saved_paths


def compare_heads_summary(attention, layer=0):
    """
    각 헤드의 Attention 분포 통계를 출력합니다.
    헤드마다 집중하는 패턴이 다른지 확인할 수 있습니다.

    Args:
        attention : [n_heads, src_len, trg_len]
        layer     : 레이어 번호 (출력용)
    """
    n_heads = attention.size(0)
    print(f"\n[Layer {layer} - 헤드별 Attention 분포 요약]")
    print(f"{'헤드':>6} | {'최대값':>8} | {'평균값':>8} | {'엔트로피':>10}")
    print("-" * 45)

    for head in range(n_heads):
        attn = attention[head].detach().cpu()

        max_val  = attn.max().item()
        mean_val = attn.mean().item()

        # 엔트로피: 높을수록 여러 토큰에 분산 / 낮을수록 특정 토큰에 집중
        entropy = -(attn * (attn + 1e-9).log()).sum(dim=-1).mean().item()

        print(f"{head+1:>6} | {max_val:>8.4f} | {mean_val:>8.4f} | {entropy:>10.4f}")

    print()
