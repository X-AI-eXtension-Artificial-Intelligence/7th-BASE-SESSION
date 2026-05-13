import torch as th
import torch.nn as nn
# ... (imports)

class YOLOv1(nn.Module):
    # 백본(Backbone) 네트워크 구성: 이미지 특징 추출 (합성곱 & 풀링 레이어)
    conv_backbone_config = [...]
    # 탐지 헤드(Detection Head) 구성: 특징을 바탕으로 바운딩 박스 예측
    conv_detection_config = [...]

    def __init__(self, S: int, B: int, C: int, mode: Optional[str] = 'detection') -> None:
        super(YOLOv1, self).__init__()
        # S: 그리드 크기 (예: 7x7)
        # B: 각 그리드 셀당 예측할 바운딩 박스 개수 (예: 2)
        # C: 클래스 개수 (예: 20)
        self.S = S
        self.B = B
        self.C = C

        # 1. 백본 네트워크 생성
        self.backbone = nn.Sequential(*backbones_modules_list)

        if mode == 'detection':
            # 2. 객체 탐지용 헤드 생성 (마지막 출력 형태: S * S * (C + B * 5))
            # 5의 의미: [x, y, w, h, confidence(객체가 있을 확률)]
            self.detection_head = nn.Sequential(detection_conv_modules, detection_fc_modules)
            self.forward = self._forward_detection
        elif mode == 'classification':
            # 사전 학습(Pretraining)용 분류 헤드
            self.classification_head = nn.Sequential(...)
            self.forward = self._forward_classification

    def _forward_detection(self, x: th.Tensor) -> th.Tensor:
        x = self.backbone(x)
        x = self.detection_head(x)
        # 최종 출력 텐서의 모양을 (Batch, S, S, C + B*5) 형태로 재배열
        y = x.reshape(x.shape[0], self.S, self.S, self.C + self.B * 5)
        return y