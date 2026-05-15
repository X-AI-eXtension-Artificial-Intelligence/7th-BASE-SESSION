import torch
from util import nms

def test(model, test_loader):
    model.eval()
    for x, y in test_loader:
        with torch.no_grad():
            predictions = model(x)
            # predictions -> bboxes 변환 후 nms 적용
            # cv2.rectangle로 이미지에 그리기
            break