import os
import matplotlib.pyplot as plt


def plot_losses(d_losses, g_losses, output_dir):
    """
    D_loss 와 G_loss 를 같은 그래프에 그려서 저장.
    정상 학습이라면:
      - D_loss ≈ 0.6~1.4 사이에서 진동
      - G_loss ≈ 1.0~3.0 사이에서 진동
    D_loss → 0 이면 D가 너무 강한 것 (Dropout 추가 또는 lr 낮추기)
    G_loss → 발산이면 Mode collapse 의심
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(d_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, d_losses, label="D Loss", color="steelblue")
    plt.plot(epochs, g_losses, label="G Loss", color="tomato")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss (Goodfellow 2014)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()
    print(f"Loss curve saved to {output_dir}/loss_curve.png")
