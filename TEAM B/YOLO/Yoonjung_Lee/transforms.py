import torch as th
from torch.nn.functional import one_hot
import torchvision.transforms.functional as fT
from PIL.Image import Image
from typing import Tuple, Optional, List

# 이미지 크기 조정 및 경계 상자 좌표 스케일링 수행 클래스
class Resize:
    """
    입력 이미지를 (D x D) 크기로 리사이즈하고 박스 좌표를 비율에 맞춰 업데이트하는 과정
    """
    def __init__(self, output_size: int) -> None:
        self.d = output_size

    def __call__(self, sample: Tuple[Image, th.Tensor]) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        img, target = sample
        w, h = img.size
        # 설정된 출력 크기로 이미지 리사이즈 수행
        img = fT.resize(img, (self.d, self.d))
        # 원본 대비 리사이즈 비율을 계산하여 좌표(xmin, xmax, ymin, ymax)에 곱하는 연산 수행
        target[:, [1, 3]] *= self.d / w
        target[:, [2, 4]] *= self.d / h
        mask = [(0, self.d), (0, self.d)]
        return img, mask, target

# 무작위 크기 조정 및 이동을 통한 데이터 증강 수행 클래스
class RandomScaleTranslate:
    """
    Resize, Zoom-out, Zoom-in 중 하나를 랜덤하게 선택하여 이미지 다양성을 확보하는 과정
    """
    def __init__(self, output_size: int, jitter: float, resize_p: float, zoom_out_p: float, zoom_in_p: float) -> None:
        self.d = output_size
        self.jitter = jitter
        self.t_probs = th.cumsum(th.Tensor([resize_p, zoom_out_p, zoom_in_p]), dim=0)

    def __call__(self, sample: Tuple[Image, th.Tensor]) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        transform_prob = th.rand(1)
        # 확률에 따라 세 가지 변환 기법 중 하나를 결정하는 분기 수행
        if transform_prob < self.t_probs[0]:
            img, mask, target = self._resize(sample)
        elif transform_prob < self.t_probs[1]:
            img, mask, target = self._zoom_out(sample)
        else:
            img, mask, target = self._zoom_in(sample)

        # 변환 후 크기가 너무 작아진 박스를 제거하여 학습 품질을 유지하는 필터링 작업
        bboxes_w = target[:, 3] - target[:, 1]
        bboxes_h = target[:, 4] - target[:, 2]
        threshold = 0.001 * self.d
        valid_bboxes = th.logical_not(th.logical_or(bboxes_w < threshold, bboxes_h < threshold))
        target = target[valid_bboxes]
        return img, mask, target

    def _zoom_out(self, sample: Tuple[Image, th.Tensor]) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        이미지 크기를 줄이고 빈 공간을 패딩으로 채워 멀리 있는 사물을 묘사하는 증강 과정
        """
        img, target = sample
        w, h = img.size
        # 무작위 종횡비 및 크기 결정 수행
        dw, dh = w * self.jitter, h * self.jitter
        rand_w, rand_h = w + th.Tensor(1).uniform_(-dw, dw), h + th.Tensor(1).uniform_(-dh, dh)
        new_ar = rand_w / rand_h
        # 리사이즈 및 패딩 위치(dx, dy) 무작위 산출 작업
        if new_ar < 1: nh = self.d; nw = int(nh * new_ar + 0.5)
        else: nw = self.d; nh = int(nw / new_ar + 0.5)
        dx, dy = th.randint(0, self.d - nw + 1, (1,)).item(), th.randint(0, self.d - nh + 1, (1,)).item()
        # 이미지 변환 및 좌표 이동 연산 수행
        img = fT.resize(img, (nh, nw))
        target[:, [1, 3]] *= nw / w; target[:, [2, 4]] *= nh / h
        img = fT.pad(img, [dx, dy, self.d - nw - dx, self.d - nh - dy])
        target[:, [1, 3]] += dx; target[:, [2, 4]] += dy
        return img, [(dx, dx + nw), (dy, dy + nh)], target

# 색상 변형(Hue, Saturation, Exposure)을 통한 증강 수행 클래스
class RandomColorJitter:
    """
    HSV 색 공간에서 색상, 채도, 명도를 무작위로 조절하여 조명 환경 변화를 묘사하는 과정
    """
    def __init__(self, hue: float, sat: float, exp: float):
        self.hue, self.sat, self.exp = hue, sat, exp

    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        # 변형 수치 무작위 샘플링 및 HSV 텐서 변환 수행
        rand_hue = th.Tensor(1).uniform_(-self.hue, self.hue)
        rand_sat = th.Tensor(1).uniform_(1 / self.sat, self.sat)
        rand_exp = th.Tensor(1).uniform_(1 / self.exp, self.exp)
        rgb_img, mask, target = sample
        hsv_tensor = fT.to_tensor(rgb_img.convert('HSV'))
        # 마스크 영역 내부의 픽셀 수치 조절 연산 수행
        mask_x, mask_y = mask
        masked_hsv = hsv_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]]
        masked_hsv[0] = (masked_hsv[0] + rand_hue) % 1.0 # 색상 조절
        masked_hsv[1] = (masked_hsv[1] * rand_sat).clamp(max=1.0) # 채도 조절
        masked_hsv[2] = (masked_hsv[2] * rand_exp).clamp(max=1.0) # 명도 조절
        return fT.to_pil_image(hsv_tensor, mode='HSV').convert('RGB'), mask, target

# 이미지 좌우 반전 및 박스 좌표 대칭 이동 수행 클래스
class RandomHorizontalFlip:
    """
    확률에 따라 이미지를 좌우 대칭시키고 x좌표를 반전시켜 데이터 양을 늘리는 과정
    """
    def __init__(self, p: float): self.p = p
    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        if th.rand(1) >= self.p: return sample
        img, mask, target = sample
        w = img.size[0]
        # x좌표 대칭 이동 및 이미지 반전 연산 수행
        target[:, [1, 3]] = w - target[:, [3, 1]]
        img = fT.hflip(img)
        # 마스크의 가로 범위도 대칭 업데이트 수행
        mask[0] = (w - mask[0][1], w - mask[0][0])
        return img, mask, target

# 일반 박스 데이터를 YOLOv1 전용 텐서 구조로 변환하는 핵심 클래스
class ToYOLOTensor:
    """
    이미지를 7x7 격자로 나누어 각 셀에 물체 유무, 클래스, 정규화된 좌표를 할당하는 과정
    """
    def __init__(self, S: int, C: int, normalize: Optional[List] = None):
        self.S, self.C, self.normalize = S, C, normalize

    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        img, mask, target = sample
        w, h = img.size
        # 격자 한 칸의 크기(cell_w, cell_h) 산출 수행
        cell_w, cell_h = w / self.S, h / self.S
        # 박스의 중심점(x, y) 및 크기(w, h) 계산 과정
        center_x, center_y = (target[:, 1] + target[:, 3]) / 2, (target[:, 2] + target[:, 4]) / 2
        bndbox_w, bndbox_h = target[:, 3] - target[:, 1], target[:, 4] - target[:, 2]
        # 중심점이 속한 격자 인덱스(row, col) 결정 수행
        center_col, center_row = th.div(center_x, cell_w, rounding_mode="trunc").long(), th.div(center_y, cell_h, rounding_mode="trunc").long()
        # 격자 내 상대 위치 및 전체 크기 대비 비율로 좌표 정규화 수행
        norm_x, norm_y = (center_x % cell_w) / cell_w, (center_y % cell_h) / cell_h
        norm_w, norm_h = bndbox_w / w, bndbox_h / h
        # (S, S, C+5) 형태의 타겟 텐서 조립 및 데이터 채우기 작업
        yolo_target = th.zeros((self.S, self.S, self.C + 5))
        yolo_target[center_row, center_col, :] = th.cat([th.ones((target.shape[0], 1)), one_hot(target[:, 0].long(), self.C),
                                                        norm_x.unsqueeze(1), norm_y.unsqueeze(1), norm_w.unsqueeze(1), norm_h.unsqueeze(1)], dim=1)
        # 이미지 텐서 변환 및 선택적 정규화 작업 수행
        img_tensor = fT.to_tensor(img)
        if self.normalize:
            mx, my = mask
            fT.normalize(img_tensor[:, my[0]:my[1], mx[0]:mx[1]], mean=self.normalize[0], std=self.normalize[1], inplace=True)
        return img_tensor, yolo_target
