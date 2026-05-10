import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import UNet


def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        x = image_tensor.unsqueeze(0).to(device)      # (1, C, H, W)
        pred = model(x)                                # (1, num_classes, H, W)
        mask = pred.argmax(dim=1).squeeze(0).cpu()     # (H, W)
        prob = torch.softmax(pred, dim=1)[0, 1].cpu()  # 병변 확률맵
    return mask, prob


def load_image(image_path, img_size=128, in_channels=3):
    """
    사용자 이미지를 로드하여 모델 입력 형식으로 변환
    """
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    tensor = transform(image)                          # (3, H, W)

    if in_channels == 1:
        tensor = tensor.mean(dim=0, keepdim=True)      # grayscale

    return tensor, image


def visualize_custom(original_image, image_tensor,
                     pred_mask, prob_map,
                     save_path="etc/custom_result.png"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 원본 이미지
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis('off')

    # 예측 마스크
    axes[1].imshow(pred_mask.numpy(), cmap='gray')
    axes[1].set_title("Predicted Mask\n(White = Lesion)", fontsize=13)
    axes[1].axis('off')

    # 확률 히트맵 (어느 부분이 병변인지)
    im = axes[2].imshow(prob_map.numpy(), cmap='hot', vmin=0, vmax=1)
    axes[2].set_title("Lesion Probability Map", fontsize=13)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    # 범례
    bg_patch = mpatches.Patch(color='black', label='Background (0)')
    fg_patch = mpatches.Patch(color='white', label='Lesion (1)')
    fig.legend(handles=[bg_patch, fg_patch],
               loc='lower center', ncol=2, fontsize=11)

    plt.suptitle("UNet Segmentation Result", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"결과 저장 완료 → {save_path}")


def compute_stats(pred_mask, img_size=128):
    """
    예측 마스크 통계 출력
    """
    total_pixels = img_size * img_size
    lesion_pixels = (pred_mask == 1).sum().item()
    bg_pixels = (pred_mask == 0).sum().item()
    lesion_ratio = lesion_pixels / total_pixels * 100

    print("\n예측 결과 통계:")
    print(f"  전체 픽셀:    {total_pixels:,}")
    print(f"  배경 픽셀:    {bg_pixels:,} ({100 - lesion_ratio:.1f}%)")
    print(f"  병변 픽셀:    {lesion_pixels:,} ({lesion_ratio:.1f}%)")
    print(f"  병변 비율:    {lesion_ratio:.2f}%")


if __name__ == "__main__":

    # -------------------------------------------------------
    # 여기에 이미지 경로 입력
    IMAGE_PATH = "dataset/test.jpg"     # 원하는 이미지 경로로 변경
    # -------------------------------------------------------

    IN_CHANNELS  = 3
    NUM_CLASSES  = 2
    IMG_SIZE     = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = UNet(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        features=[64, 128, 256, 512]
    ).to(device)
    model.load_state_dict(
        torch.load("etc/unet_best.pt", map_location=device)
    )
    print("모델 로드 완료")

    # 이미지 로드
    image_tensor, original_image = load_image(
        IMAGE_PATH, img_size=IMG_SIZE, in_channels=IN_CHANNELS
    )
    print(f"이미지 로드 완료: {IMAGE_PATH}")
    print(f"입력 크기: {image_tensor.shape}")

    # 예측
    pred_mask, prob_map = predict(model, image_tensor, device)

    # 통계 출력
    compute_stats(pred_mask, IMG_SIZE)

    # 시각화
    visualize_custom(original_image, image_tensor, pred_mask, prob_map)