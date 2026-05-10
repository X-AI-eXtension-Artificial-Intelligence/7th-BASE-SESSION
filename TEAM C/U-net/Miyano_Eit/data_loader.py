import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticSegDataset(Dataset):
    """
    의료 영상 분할 toy 데이터셋
    배경(0)과 병변(1)으로 구성된 합성 이미지

    실제 의료 데이터 사용 시:
    - 이미지: .png/.jpg
    - 마스크: 픽셀별 클래스 레이블 (0, 1, 2, ...)
    """
    def __init__(self, num_samples=500, img_size=128,
                 num_classes=2, in_channels=3):
        self.num_classes = num_classes
        print("데이터 미리 생성 중...")

        self.images = torch.zeros(num_samples, in_channels, img_size, img_size)
        self.masks = torch.zeros(num_samples, img_size, img_size, dtype=torch.long)

        for idx in range(num_samples):
            img = torch.ones(in_channels, img_size, img_size) * 0.2

            cx = torch.randint(img_size // 4, 3 * img_size // 4, (1,)).item()
            cy = torch.randint(img_size // 4, 3 * img_size // 4, (1,)).item()
            r = torch.randint(10, img_size // 4, (1,)).item()

            # 이중 for loop 대신 meshgrid로 한번에 계산
            y_grid, x_grid = torch.meshgrid(
            torch.arange(img_size),
            torch.arange(img_size),
            indexing='ij'
    )
            circle = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) < r ** 2

            mask = circle.long()
            img[:, circle] = 0.9

            self.images[idx] = img
            self.masks[idx] = mask

        print(f"데이터 생성 완료: {num_samples}장")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


def get_dataloader(num_samples=500, img_size=128, num_classes=2,
                   in_channels=3, batch_size=8, shuffle=True):
    dataset = SyntheticSegDataset(num_samples, img_size, num_classes, in_channels)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)