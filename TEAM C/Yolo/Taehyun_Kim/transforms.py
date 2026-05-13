import torch as th
from torch.nn.functional import one_hot
import torchvision.transforms.functional as fT
from PIL.Image import Image
from typing import Tuple, Optional, List



# Resize: 이미지를 정사각형(d×d)으로 리사이즈하고
#         바운딩 박스 좌표를 비율에 맞게 스케일링
class Resize:

    def __init__(self, output_size: int) -> None:
        # 출력 이미지의 한 변 크기 저장
        self.d = output_size

    def __call__(self, sample: Tuple[Image, th.Tensor]):
        """
        (h×w) 이미지를 (d×d)로 변환하고 바운딩 박스 좌표를 스케일링.
          x' = x * d / w
          y' = y * d / h

        반환: (리사이즈된 이미지, 마스크, 변환된 타겟)
          마스크: 이미지 유효 영역 [(x_start, x_end), (y_start, y_end)]
                 Resize는 패딩 없이 전체가 유효하므로 [(0,d), (0,d)]
        """
        img, target = sample
        w, h = img.size

        # 이미지를 d×d로 리사이즈
        img = fT.resize(img, (self.d, self.d))
        # x 좌표 (xmin, xmax) 스케일링
        target[:, [1, 3]] *= self.d / w
        # y 좌표 (ymin, ymax) 스케일링
        target[:, [2, 4]] *= self.d / h

        # 마스크: 패딩 없이 전체 이미지가 유효
        mask = [(0, self.d), (0, self.d)]
        return img, mask, target


# RandomScaleTranslate: 세 가지 방식 중 하나를 랜덤 선택해 이미지 변환
#   1. resize       : 단순 리사이즈
#   2. zoom out     : 이미지를 줄이고 제로 패딩 (확대 시야)
#   3. zoom in      : 이미지 일부를 잘라서 리사이즈 (확대)
class RandomScaleTranslate:

    def __init__(self, output_size, jitter, resize_p, zoom_out_p, zoom_in_p):
        """
        output_size : 출력 이미지 크기 (d×d)
        jitter      : 랜덤 스케일/이동의 강도
        resize_p    : 단순 리사이즈 확률
        zoom_out_p  : Zoom out 확률
        zoom_in_p   : Zoom in 확률
        """
        self.d = output_size
        self.jitter = jitter
        # 누적 확률로 저장 (th.rand와 비교하기 위해)
        self.t_probs = th.cumsum(th.Tensor([resize_p, zoom_out_p, zoom_in_p]), dim=0)

    def __call__(self, sample: Tuple[Image, th.Tensor]):
        # 균일 분포에서 샘플링해 어떤 변환을 적용할지 결정
        transform_prob = th.rand(1)
        if transform_prob < self.t_probs[0]:       # resize 선택
            img, mask, target = self._resize(sample)
        elif transform_prob < self.t_probs[1]:     # zoom out 선택
            img, mask, target = self._zoom_out(sample)
        else:                                      # zoom in 선택
            img, mask, target = self._zoom_in(sample)

        # 너무 작은 바운딩 박스 제거 (이미지 크기의 0.1% 미만)
        bboxes_w = target[:, 3] - target[:, 1]
        bboxes_h = target[:, 4] - target[:, 2]
        threshold = 0.001 * self.d
        valid_bboxes = th.logical_not(th.logical_or(bboxes_w < threshold, bboxes_h < threshold))
        target = target[valid_bboxes]
        return img, mask, target

    def _resize(self, sample):
        """단순 리사이즈: Resize 클래스와 동일한 로직"""
        img, target = sample
        w, h = img.size
        img = fT.resize(img, (self.d, self.d))
        target[:, [1, 3]] *= self.d / w
        target[:, [2, 4]] *= self.d / h
        mask = [(0, self.d), (0, self.d)]
        return img, mask, target

    def _zoom_out(self, sample):
        """
        Zoom Out: 이미지를 랜덤 비율로 줄인 후 제로 패딩으로 d×d로 맞춤.
          1. jitter 범위 내에서 새로운 종횡비(new_ar) 샘플링
          2. 긴 변을 d에 맞추고 짧은 변은 종횡비에 맞게 결정
          3. 이미지 패치를 d×d 캔버스에 랜덤 위치에 배치 (나머지는 0으로 패딩)
          → 좌표도 리사이즈 + 이동 보정
          → 마스크는 패딩이 없는 영역(실제 이미지 영역)을 나타냄
        """
        img, target = sample
        w, h = img.size

        # 랜덤 종횡비 결정
        dw = w * self.jitter
        dh = h * self.jitter
        rand_w = w + th.Tensor(1).uniform_(-dw, dw)
        rand_h = h + th.Tensor(1).uniform_(-dh, dh)
        new_ar = rand_w / rand_h

        # 긴 변이 d가 되도록 새로운 크기 결정
        if new_ar < 1:   # 세로가 더 길면
            nh = self.d
            nw = int(nh * new_ar + 0.5)
        else:            # 가로가 더 길면
            nw = self.d
            nh = int(nw / new_ar + 0.5)

        # 패딩 시작점(dx, dy) 랜덤 결정
        dx = th.randint(low=0, high=self.d - nw + 1, size=(1,)).item()
        dy = th.randint(low=0, high=self.d - nh + 1, size=(1,)).item()

        # 이미지 리사이즈 후 패딩 적용
        img = fT.resize(img, (nh, nw))
        target[:, [1, 3]] *= nw / w    # x 좌표 리사이즈 반영
        target[:, [2, 4]] *= nh / h    # y 좌표 리사이즈 반영
        img = fT.pad(img, padding=[dx, dy, self.d - nw - dx, self.d - nh - dy])
        target[:, [1, 3]] += dx        # 패딩(이동) 반영
        target[:, [2, 4]] += dy

        mask = [(dx, dx + nw), (dy, dy + nh)]  # 실제 이미지 영역
        return img, mask, target

    def _zoom_in(self, sample):
        """
        Zoom In: 원본 이미지의 일부(patch)를 잘라서 d×d로 확대.
          1. nw ~ U((1-jitter)w, w), nh ~ U((1-jitter)h, h) 로 패치 크기 결정
          2. dx, dy로 패치의 시작 위치 결정
          3. 패치를 d×d로 리사이즈
          → 가시 범위 밖 박스 제거, 부분 가시 박스는 좌표 클램핑
        """
        img, target = sample
        w, h = img.size

        # 패치 크기 랜덤 샘플링
        nw = int(th.Tensor(1).uniform_((1 - self.jitter) * w, w) + 0.5)
        nh = int(th.Tensor(1).uniform_((1 - self.jitter) * h, h) + 0.5)
        # 패치 시작 위치 랜덤 샘플링
        dx = int(th.Tensor(1).uniform_(0, w - nw + 1) + 0.5)
        dy = int(th.Tensor(1).uniform_(0, h - nh + 1) + 0.5)

        # 패치를 잘라서 d×d로 리사이즈
        img = fT.resized_crop(img, top=dy, left=dx, height=nh, width=nw, size=(self.d, self.d))

        # 좌표 변환: 이동(패치 시작점 기준) → 리사이즈
        target[:, [1, 3]] -= dx
        target[:, [2, 4]] -= dy
        target[:, [1, 3]] *= self.d / nw
        target[:, [2, 4]] *= self.d / nh

        # 이미지 밖으로 완전히 나간 박스 제거
        target = target[th.logical_not(
            th.logical_or(
                th.logical_or(target[:, 3] < 0, target[:, 1] > self.d),
                th.logical_or(target[:, 4] < 0, target[:, 2] > self.d)
            )
        )]
        # 부분적으로 가시인 박스는 이미지 경계에 클램핑
        target[:, [1, 2]] = target[:, [1, 2]].clamp(min=0)
        target[:, [3, 4]] = target[:, [3, 4]].clamp(max=self.d)

        mask = [(0, self.d), (0, self.d)]  # Zoom in은 전체가 유효
        return img, mask, target


# RandomColorJitter: HSV 색공간에서 색상(Hue), 채도(Saturation),
#                    노출(Exposure/Value)을 랜덤하게 조정

class RandomColorJitter:

    def __init__(self, hue: float, sat: float, exp: float):
        """
        hue : 색조 변화 범위 ([-hue, hue]에서 균일 샘플링)
        sat : 채도 변화 범위 ([1/sat, sat]에서 균일 샘플링)
        exp : 노출 변화 범위 ([1/exp, exp]에서 균일 샘플링)
        """
        self.hue = hue
        self.sat = sat
        self.exp = exp

    def __call__(self, sample):
        # 이 이미지에 적용할 랜덤 값 샘플링
        rand_hue = th.Tensor(1).uniform_(-self.hue, self.hue)
        rand_sat = th.Tensor(1).uniform_(1 / self.sat, self.sat)
        rand_exp = th.Tensor(1).uniform_(1 / self.exp, self.exp)

        rgb_img, mask, target = sample
        # RGB → HSV 변환 (PIL 내장)
        hsv_img = rgb_img.convert('HSV')
        hsv_tensor = fT.to_tensor(hsv_img)  # 값 범위: [0, 1]

        # 마스크 영역(실제 이미지, 패딩 제외)에만 색상 조정 적용
        mask_x, mask_y = mask
        masked_hsv_tensor = hsv_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]]

        # H(색조) 조정: 순환 특성 고려 (1 초과 → -1, 0 미만 → +1)
        masked_hsv_tensor[0] += rand_hue
        masked_hsv_tensor[0] += (1. * (masked_hsv_tensor[0] < 0) - 1. * (masked_hsv_tensor[0] > 1))

        # S(채도) 조정: 곱셈 후 [0, 1] 클램핑
        masked_hsv_tensor[1] *= rand_sat
        masked_hsv_tensor[1] = masked_hsv_tensor[1].clamp(max=1.0)

        # V(노출/밝기) 조정: 곱셈 후 [0, 1] 클램핑
        masked_hsv_tensor[2] *= rand_exp
        masked_hsv_tensor[2] = masked_hsv_tensor[2].clamp(max=1.0)

        # HSV Tensor → HSV PIL Image → RGB PIL Image
        hsv_img = fT.to_pil_image(hsv_tensor, mode='HSV')
        rgb_img = hsv_img.convert('RGB')

        return rgb_img, mask, target



# RandomHorizontalFlip: 확률 p로 이미지를 수평 반전
#   반전 시 바운딩 박스 x 좌표도 함께 변환
class RandomHorizontalFlip:

    def __init__(self, p: float) -> None:
        self.p = p  # 반전 적용 확률

    def __call__(self, sample):
        # p 확률로 반전 적용
        apply_transform = th.rand(1) < self.p
        if not apply_transform:
            return sample

        img, mask, target = sample
        w = img.size[0]

        # 바운딩 박스 x 좌표 반전: xmin' = w - xmax, xmax' = w - xmin
        target[:, [1, 3]] = w - target[:, [3, 1]]
        img = fT.hflip(img)

        # 마스크의 x 범위도 반전
        start_x, end_x = mask[0]
        mask[0] = (w - end_x, w - start_x)

        return img, mask, target


# ToYOLOTensor: PIL Image와 박스 좌표를 YOLO 학습용 Tensor로 변환
#   - 이미지: PIL Image → 정규화된 Float Tensor
#   - 타겟 : (N, 5) → (S, S, C+5) YOLO 그리드 형식

class ToYOLOTensor:

    def __init__(self, S: int, C: int, normalize: Optional[List] = None) -> None:
        """
        S         : 그리드 분할 수 (이미지를 S×S로 나눔)
        C         : 클래스 수
        normalize : [[mean_R, mean_G, mean_B], [std_R, std_G, std_B]] (없으면 정규화 안함)
        """
        self.S = S
        self.C = C
        self.normalize = normalize

    def __call__(self, sample):
        """
        YOLO 그리드 타겟 형식 (S, S, C+5):
          [0]       : 해당 셀에 객체 존재 여부 (0 or 1)
          [1:C+1]   : 클래스 원-핫 인코딩
          [C+1]     : 셀 내 중심 x 좌표 (셀 크기로 정규화, 0~1)
          [C+2]     : 셀 내 중심 y 좌표 (셀 크기로 정규화, 0~1)
          [C+3]     : 바운딩 박스 너비 (이미지 너비로 정규화, 0~1)
          [C+4]     : 바운딩 박스 높이 (이미지 높이로 정규화, 0~1)
        """
        img, mask, target = sample
        w, h = img.size

        # 셀 하나의 픽셀 크기
        cell_w = w / self.S
        cell_h = h / self.S

        # 바운딩 박스 중심 좌표 계산
        center_x = (target[:, 1] + target[:, 3]) / 2
        center_y = (target[:, 2] + target[:, 4]) / 2
        # 바운딩 박스 너비/높이
        bndbox_w = target[:, 3] - target[:, 1]
        bndbox_h = target[:, 4] - target[:, 2]

        label = target[:, 0].long()
        # 중심이 속하는 그리드 셀의 열(col)/행(row) 인덱스
        center_col = th.div(center_x, cell_w, rounding_mode="trunc").long()
        center_row = th.div(center_y, cell_h, rounding_mode="trunc").long()
        # 셀 기준 정규화 좌표 (셀 내에서의 상대 위치)
        norm_center_x = (center_x % cell_w) / cell_w
        norm_center_y = (center_y % cell_h) / cell_h
        # 이미지 전체 크기 기준 정규화 (0~1)
        norm_bndbox_w = bndbox_w / w
        norm_bndbox_h = bndbox_h / h

        # (S, S, C+5) 빈 타겟 텐서 생성 (기본값 0)
        target = th.zeros((self.S, self.S, self.C + 5))
        # 객체가 있는 셀에 값 할당: [존재플래그, 클래스원핫, x, y, w, h]
        target[center_row, center_col, :] = th.cat([
            th.ones((label.shape[0], 1)),      # 객체 존재 플래그
            one_hot(label, self.C),             # 클래스 원-핫
            norm_center_x.unsqueeze(1),
            norm_center_y.unsqueeze(1),
            norm_bndbox_w.unsqueeze(1),
            norm_bndbox_h.unsqueeze(1)
        ], dim=1)

        # PIL Image → [0,1] Float Tensor (C, H, W)
        img_tensor = fT.to_tensor(img)
        # 정규화: 마스크 영역(실제 이미지 부분)에만 적용 (패딩 픽셀은 제외)
        if self.normalize:
            mask_x, mask_y = mask
            fT.normalize(
                img_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]],
                mean=self.normalize[0],
                std=self.normalize[1],
                inplace=True
            )

        return img_tensor, target



# ImgToTensor: evaluate.py에서 사용. 이미지만 Tensor로 변환 (타겟은 유지)
#   학습용 ToYOLOTensor와 달리 타겟을 YOLO 그리드로 변환하지 않는다.
class ImgToTensor:

    def __init__(self, normalize: Optional[List] = None) -> None:
        self.normalize = normalize

    def __call__(self, sample):
        img, mask, target = sample

        # PIL Image → Float Tensor
        img_tensor = fT.to_tensor(img)
        # 마스크 영역에만 정규화 적용
        if self.normalize:
            mask_x, mask_y = mask
            img_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]] = fT.normalize(
                img_tensor,
                mean=self.normalize[0],
                std=self.normalize[1]
            )
        return img_tensor, target
