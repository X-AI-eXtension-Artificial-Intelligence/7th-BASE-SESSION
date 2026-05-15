import torch
import torch.optim as optim
from model import YOLOv1
from data import YOLODataset
from util import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S, self.B, self.C = S, B, C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        _, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)

        # 1. Coordinate Loss
        box_predictions = exists_box * (bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25])
        box_targets = exists_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        loss_coord = self.mse(torch.flatten(box_predictions, end_dim=-2), torch.flatten(box_targets, end_dim=-2))

        # 2. Object Loss / 3. No Object Loss / 4. Class Loss 생략 (구조 동일)
        # ... (이하 YOLO Loss 공식에 따른 구현) ...
        return self.lambda_coord * loss_coord # + 나머지 losses

# 실제 학습 루프
def main():
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20)
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.0005)
    loss_fn = YoloLoss()
    # 데이터 로더 설정 및 학습 진행