import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SyntheticYOLODataset(Dataset):
    """
    실제 이미지 없이 색깔 박스로 합성 이미지를 만드는 데이터셋
    색깔 박스 = 객체라고 가르침
    """
    def __init__(self, num_samples=1000, S=7, B=2, C=20, img_size=224):
        self.num_samples = num_samples
        self.S = S
        self.B = B
        self.C = C
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 회색 배경 이미지 생성
        image = torch.ones(3, self.img_size, self.img_size) * 0.5

        target = torch.zeros(self.S, self.S, self.B * 5 + self.C)

        # 랜덤 객체 1~3개 생성
        num_objects = torch.randint(1, 4, (1,)).item()

        for _ in range(num_objects):
            # 랜덤 클래스 (여기서는 car=6만 사용)
            class_id = 6

            # 랜덤 위치와 크기
            cx = torch.rand(1).item()           # 0~1
            cy = torch.rand(1).item()
            w = torch.rand(1).item() * 0.3 + 0.1    # 0.1~0.4
            h = torch.rand(1).item() * 0.3 + 0.1

            # 이미지에 색깔 박스 그리기 (빨간색)
            x1 = int(max(0, (cx - w/2)) * self.img_size)
            y1 = int(max(0, (cy - h/2)) * self.img_size)
            x2 = int(min(1, (cx + w/2)) * self.img_size)
            y2 = int(min(1, (cy + h/2)) * self.img_size)

            image[0, y1:y2, x1:x2] = 0.9     # R 채널 높임 (빨간색)
            image[1, y1:y2, x1:x2] = 0.1     # G 채널 낮춤
            image[2, y1:y2, x1:x2] = 0.1     # B 채널 낮춤

            # 어느 그리드 셀에 해당하는지
            gi = int(cy * self.S)
            gj = int(cx * self.S)
            gi = min(gi, self.S - 1)
            gj = min(gj, self.S - 1)

            # 셀 내 상대 좌표
            x_cell = cx * self.S - gj
            y_cell = cy * self.S - gi

            if target[gi, gj, 4] == 0:        # 아직 객체 없는 셀만
                target[gi, gj, 0:5] = torch.tensor(
                    [x_cell, y_cell, w, h, 1.0]
                )
                target[gi, gj, 5:10] = torch.tensor(
                    [x_cell, y_cell, w, h, 1.0]
                )
                target[gi, gj, self.B * 5 + class_id] = 1.0

        return image, target


def get_synthetic_dataloader(num_samples=1000, S=7, B=2, C=20,
                             batch_size=16, shuffle=True, img_size=224):
    dataset = SyntheticYOLODataset(num_samples=num_samples,
                                   S=S, B=B, C=C, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)