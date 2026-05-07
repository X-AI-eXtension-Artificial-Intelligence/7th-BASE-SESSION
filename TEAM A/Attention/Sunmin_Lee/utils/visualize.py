import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch


def plot_attention(src_seq, tgt_seq, attention_weights, save_path="attention_map.png"):
    """
    Attention weight 히트맵 시각화

    Args:
        src_seq         : 소스 시퀀스 (list)
        tgt_seq         : 타깃 시퀀스 (list)
        attention_weights: (tgt_len, src_len) tensor or numpy
        save_path       : 저장 경로
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 7))

    im = ax.imshow(attention_weights, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(src_seq)))
    ax.set_yticks(range(len(tgt_seq)))
    ax.set_xticklabels(src_seq, fontsize=13)
    ax.set_yticklabels(tgt_seq, fontsize=13)

    ax.set_xlabel("Source", fontsize=14)
    ax.set_ylabel("Target (predicted)", fontsize=14)
    ax.set_title("Bahdanau Attention Weights", fontsize=15)

    # 각 셀에 수치 표시
    for i in range(len(tgt_seq)):
        for j in range(len(src_seq)):
            ax.text(j, i, f"{attention_weights[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if attention_weights[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"저장 완료: {save_path}")


def print_prediction(src_seq, tgt_seq, pred_seq):
    """
    예측 결과 콘솔 출력

    Args:
        src_seq  : 소스 시퀀스 (list)
        tgt_seq  : 정답 시퀀스 (list)
        pred_seq : 예측 시퀀스 (list)
    """
    print("=" * 40)
    print(f"Source : {src_seq}")
    print(f"Target : {tgt_seq}")
    print(f"Predict: {pred_seq}")
    correct = tgt_seq == pred_seq
    print(f"Result : {'✓ 정답' if correct else '✗ 오답'}")
    print("=" * 40)