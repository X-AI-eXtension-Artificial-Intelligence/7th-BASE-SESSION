"""
transforms.py - 데이터 증강 및 전처리 변환 클래스

YOLOv1 학습을 위한 이미지 변환 파이프라인을 정의합니다.
주요 변환:
  - Resize: 이미지를 고정 크기로 리사이즈
  - RandomScaleTranslate: 랜덤 스케일/이동 (zoom in/out)
  - RandomColorJitter: HSV 색상 공간에서 랜덤 색상 왜곡
  - RandomHorizontalFlip: 랜덤 좌우 반전
  - ToYOLOTensor: YOLO 그리드 형식으로 타겟 변환 + 텐서 변환
  - ImgToTensor: 단순 이미지→텐서 변환 (사전학습용)
"""

import torch as th
from torch.nn.functional import one_hot
import torchvision.transforms.functional as fT
from PIL.Image import Image
from typing import Tuple, Optional, List


class Resize:
    """
    이미지를 (d × d) 크기로 리사이즈하고, 바운딩 박스 좌표를 비례적으로 스케일링합니다.
    """

    def __init__(self, output_size: int) -> None:
        """
        :param output_size: 변환 후 이미지 크기 (d × d)
        """
        self.d = output_size

    def __call__(self, sample: Tuple[Image, th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        이미지 리사이즈 및 바운딩 박스 좌표 변환.
        좌표 변환: x' = x * d / w,  y' = y * d / h

        :param sample: (이미지, 타겟) 튜플
        :return: (리사이즈된 이미지, 마스크, 변환된 타겟)
        """
        img, target = sample
        w, h = img.size

        img = fT.resize(img, (self.d, self.d))
        target[:, [1, 3]] *= self.d / w  # x 좌표 스케일링
        target[:, [2, 4]] *= self.d / h  # y 좌표 스케일링

        # 마스크: 전체 이미지 영역 (패딩 없음)
        mask = [(0, self.d), (0, self.d)]
        return img, mask, target


class RandomScaleTranslate:
    """
    랜덤 스케일/이동 변환.
    세 가지 연산 중 하나를 확률적으로 선택:
      - resize: 단순 리사이즈
      - zoom out: 축소 후 제로 패딩으로 이동
      - zoom in: 랜덤 크롭 후 리사이즈
    """
    def __init__(self,
                 output_size: int,
                 jitter: float,
                 resize_p: float,
                 zoom_out_p: float,
                 zoom_in_p: float) -> None:
        """
        :param output_size: 출력 이미지 크기 (d × d)
        :param jitter: 스케일/이동의 랜덤 범위를 결정하는 지터 계수
        :param resize_p: 단순 리사이즈 확률
        :param zoom_out_p: 축소(zoom out) 확률
        :param zoom_in_p: 확대(zoom in) 확률
        """
        self.d = output_size
        self.jitter = jitter
        # 누적 확률로 변환하여 연산 선택에 사용
        self.t_probs = th.cumsum(th.Tensor([resize_p, zoom_out_p, zoom_in_p]), dim=0)

    def __call__(self, sample: Tuple[Image, th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        랜덤으로 변환 연산을 선택하여 적용합니다.
        변환 후 너무 작아진 바운딩 박스는 제거합니다.

        :param sample: (이미지, 타겟) 튜플
        :return: (변환된 이미지, 마스크, 변환된 타겟)
        """
        transform_prob = th.rand(1)
        if transform_prob < self.t_probs[0]:                    # 단순 리사이즈
            img, mask, target = self._resize(sample)
        elif transform_prob < self.t_probs[1]:                  # 축소 + 리사이즈
            img, mask, target = self._zoom_out(sample)
        else:                                                   # 확대 + 리사이즈
            img, mask, target = self._zoom_in(sample)

        # 너무 작은 바운딩 박스 제거 (임계값: 0.001 * d)
        bboxes_w = target[:, 3] - target[:, 1]
        bboxes_h = target[:, 4] - target[:, 2]
        threshold = 0.001 * self.d
        valid_bboxes = th.logical_not(th.logical_or(bboxes_w < threshold, bboxes_h < threshold))
        target = target[valid_bboxes]
        return img, mask, target

    def _resize(self, sample: Tuple[Image, th.Tensor]
                ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        단순 리사이즈: Resize 클래스와 동일한 로직.

        :param sample: (이미지, 타겟) 튜플
        :return: (리사이즈된 이미지, 마스크, 변환된 타겟)
        """
        img, target = sample
        w, h = img.size

        img = fT.resize(img, (self.d, self.d))
        target[:, [1, 3]] *= self.d / w
        target[:, [2, 4]] *= self.d / h

        mask = [(0, self.d), (0, self.d)]
        return img, mask, target

    def _zoom_out(self, sample: Tuple[Image, th.Tensor]
                  ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        축소(Zoom Out) 변환:
        1. 랜덤 종횡비로 이미지를 축소 (큰 축 = d, 작은 축 < d)
        2. 제로 패딩으로 랜덤 위치에 배치하여 d×d 이미지 생성
        3. 마스크로 실제 이미지 영역을 기록 (색상 왜곡 시 패딩 영역 보호용)

        :param sample: (이미지, 타겟) 튜플
        :return: (변환된 이미지, 마스크, 변환된 타겟)
        """
        img, target = sample
        w, h = img.size

        # 랜덤 종횡비 결정
        dw = w * self.jitter
        dh = h * self.jitter
        rand_w = w + th.Tensor(1).uniform_(-dw, dw)
        rand_h = h + th.Tensor(1).uniform_(-dh, dh)
        new_ar = rand_w / rand_h

        # 큰 축을 d로 설정하고 작은 축을 종횡비에 맞게 계산
        if new_ar < 1:
            nh = self.d
            nw = int(nh * new_ar + 0.5)
        else:
            nw = self.d
            nh = int(nw / new_ar + 0.5)

        # 랜덤 이동량 결정 (패딩 위치)
        dx = th.randint(low=0, high=self.d - nw + 1, size=(1,)).item()
        dy = th.randint(low=0, high=self.d - nh + 1, size=(1,)).item()

        # 리사이즈 후 좌표 변환
        img = fT.resize(img, (nh, nw))
        target[:, [1, 3]] *= nw / w
        target[:, [2, 4]] *= nh / h

        # 제로 패딩 적용 및 좌표 이동
        img = fT.pad(img, padding=[dx, dy, self.d - nw - dx, self.d - nh - dy])
        target[:, [1, 3]] += dx
        target[:, [2, 4]] += dy

        # 마스크: 실제 이미지가 있는 영역
        mask = [(dx, dx + nw), (dy, dy + nh)]
        return img, mask, target

    def _zoom_in(self, sample: Tuple[Image, th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        확대(Zoom In) 변환:
        1. 원본 이미지에서 랜덤 크기의 패치를 크롭
        2. 크롭된 패치를 d×d로 리사이즈
        3. 크롭 영역 밖의 바운딩 박스는 제거, 부분적으로 보이는 박스는 클램핑

        :param sample: (이미지, 타겟) 튜플
        :return: (변환된 이미지, 마스크, 변환된 타겟)
        """
        img, target = sample
        w, h = img.size

        # 랜덤 크롭 크기 및 위치 결정
        nw = int(th.Tensor(1).uniform_((1 - self.jitter) * w, w) + 0.5)
        nh = int(th.Tensor(1).uniform_((1 - self.jitter) * h, h) + 0.5)
        dx = int(th.Tensor(1).uniform_(0, w - nw + 1) + 0.5)
        dy = int(th.Tensor(1).uniform_(0, h - nh + 1) + 0.5)

        # 크롭 후 리사이즈
        img = fT.resized_crop(img, top=dy, left=dx, height=nh, width=nw, size=(self.d, self.d))

        # 좌표 변환: 크롭 원점 이동 → 스케일링
        target[:, [1, 3]] -= dx
        target[:, [2, 4]] -= dy
        target[:, [1, 3]] *= self.d / nw
        target[:, [2, 4]] *= self.d / nh

        # 완전히 보이지 않는 바운딩 박스 제거
        target = target[th.logical_not(th.logical_or(th.logical_or(target[:, 3] < 0, target[:, 1] > self.d),
                                                     th.logical_or(target[:, 4] < 0, target[:, 2] > self.d)))]

        # 부분적으로 보이는 바운딩 박스의 좌표를 이미지 범위로 클램핑
        target[:, [1, 2]] = target[:, [1, 2]].clamp(min=0)
        target[:, [3, 4]] = target[:, [3, 4]].clamp(max=self.d)

        mask = [(0, self.d), (0, self.d)]
        return img, mask, target


class RandomColorJitter:
    """
    랜덤 색상 왜곡 변환.
    HSV 색상 공간에서 색조(Hue), 채도(Saturation), 노출(Exposure/Value)을 랜덤으로 조절합니다.
    """

    def __init__(self, hue: float, sat: float, exp: float):
        """
        :param hue: 색조 변화 범위 [-hue, hue]
        :param sat: 채도 변화 범위 [1/sat, sat]
        :param exp: 노출 변화 범위 [1/exp, exp]
        """
        self.hue = hue
        self.sat = sat
        self.exp = exp

    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        HSV 색상 공간에서 랜덤 색상 왜곡을 적용합니다.
        마스크를 사용하여 패딩된 영역(제로 패딩)은 왜곡하지 않습니다.

        :param sample: (이미지, 마스크, 타겟) 튜플
        :return: (색상 왜곡된 이미지, 마스크, 타겟)
        """
        # 랜덤 색상 파라미터 샘플링
        rand_hue = th.Tensor(1).uniform_(-self.hue, self.hue)
        rand_sat = th.Tensor(1).uniform_(1 / self.sat, self.sat)
        rand_exp = th.Tensor(1).uniform_(1 / self.exp, self.exp)

        # RGB → HSV 변환
        rgb_img, mask, target = sample
        hsv_img = rgb_img.convert('HSV')
        hsv_tensor = fT.to_tensor(hsv_img)

        # 마스크 영역만 색상 조절 (패딩 영역 보호)
        mask_x, mask_y = mask
        masked_hsv_tensor = hsv_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]]

        # 색조(H) 조절: 값이 [0,1] 범위를 벗어나면 순환
        masked_hsv_tensor[0, :, :] += rand_hue
        masked_hsv_tensor[0, :, :] += (1. * (masked_hsv_tensor[0, :, :] < 0) - 1. * \
                                       (masked_hsv_tensor[0, :, :] > 1)) * th.ones_like(masked_hsv_tensor[0, :, :])
        # 채도(S) 조절: 최대 1.0으로 클램핑
        masked_hsv_tensor[1, :, :] *= rand_sat
        masked_hsv_tensor[1, :, :] = masked_hsv_tensor[1, :, :].clamp(max=1.0)

        # 노출/밝기(V) 조절: 최대 1.0으로 클램핑
        masked_hsv_tensor[2, :, :] *= rand_exp
        masked_hsv_tensor[2, :, :] = masked_hsv_tensor[2, :, :].clamp(max=1.0)

        # HSV → RGB 변환
        hsv_img = fT.to_pil_image(hsv_tensor, mode='HSV')
        rgb_img = hsv_img.convert('RGB')

        return rgb_img, mask, target


class RandomHorizontalFlip:
    """
    랜덤 좌우 반전 변환.
    확률 p로 이미지를 수평 반전하고, 바운딩 박스 좌표와 마스크도 함께 변환합니다.
    """
    def __init__(self, p: float) -> None:
        """
        :param p: 수평 반전 적용 확률
        """
        self.p = p

    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        확률 p로 이미지를 좌우 반전합니다.
        반전 시 xmin, xmax 좌표를 w - xmax, w - xmin으로 변환합니다.

        :param sample: (이미지, 마스크, 타겟) 튜플
        :return: (반전된 이미지, 변환된 마스크, 변환된 타겟) 또는 원본
        """
        apply_transform = th.rand(1) < self.p
        if not apply_transform:
            return sample

        img, mask, target = sample
        w = img.size[0]

        # x 좌표 반전: new_xmin = w - old_xmax, new_xmax = w - old_xmin
        target[:, [1, 3]] = w - target[:, [3, 1]]
        img = fT.hflip(img)

        # 마스크의 x 범위도 반전
        start_x, end_x = mask[0]
        mask[0] = (w - end_x, w - start_x)

        return img, mask, target


class ToYOLOTensor:
    """
    YOLO 그리드 형식으로 타겟을 변환하고, PIL 이미지를 텐서로 변환합니다.

    입력 타겟: (N, 5) - [클래스, xmin, ymin, xmax, ymax]
    출력 타겟: (S, S, C+5) - 각 셀: [존재여부, 원핫클래스(C), cx, cy, w, h]
      - cx, cy: 셀 내 상대 좌표 (0~1)
      - w, h: 이미지 크기로 정규화된 바운딩 박스 크기
    """

    def __init__(self, S: int, C: int, normalize: Optional[List] = None) -> None:
        """
        :param S: 그리드 크기 (S × S)
        :param C: 클래스 수
        :param normalize: [평균값 리스트, 표준편차 리스트] (채널별 정규화용)
        """
        self.S = S
        self.C = C
        self.normalize = normalize

    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[th.Tensor, th.Tensor]:
        """
        이미지를 텐서로 변환하고, 바운딩 박스를 YOLO 그리드 형식으로 인코딩합니다.

        YOLO 타겟 형식 (S×S×(C+5)):
          - [0]: 객체 존재 여부 (0 또는 1)
          - [1:C+1]: 클래스 원핫 인코딩
          - [C+1]: 셀 내 정규화된 중심 x좌표
          - [C+2]: 셀 내 정규화된 중심 y좌표
          - [C+3]: 이미지 대비 정규화된 바운딩 박스 너비
          - [C+4]: 이미지 대비 정규화된 바운딩 박스 높이

        :param sample: (이미지, 마스크, 타겟) 튜플
        :return: (이미지 텐서, YOLO 그리드 타겟 텐서)
        """
        img, mask, target = sample
        w, h = img.size

        # 셀 크기 계산
        cell_w = w / self.S
        cell_h = h / self.S

        # 바운딩 박스 중심 좌표 및 크기 계산
        center_x = (target[:, 1] + target[:, 3]) / 2
        center_y = (target[:, 2] + target[:, 4]) / 2
        bndbox_w = target[:, 3] - target[:, 1]
        bndbox_h = target[:, 4] - target[:, 2]

        # 중심이 속하는 그리드 셀 결정
        label = target[:, 0].long()
        center_col = th.div(center_x, cell_w, rounding_mode="trunc").long()
        center_row = th.div(center_y, cell_h, rounding_mode="trunc").long()

        # 셀 내 상대 좌표로 정규화 (0~1)
        norm_center_x = (center_x % cell_w) / cell_w
        norm_center_y = (center_y % cell_h) / cell_h
        # 이미지 크기로 정규화
        norm_bndbox_w = bndbox_w / w
        norm_bndbox_h = bndbox_h / h

        # S×S×(C+5) 그리드 타겟 생성
        target = th.zeros((self.S, self.S, self.C + 5))
        target[center_row, center_col, :] = th.cat([th.ones((label.shape[0], 1)),
                                                    one_hot(label, self.C),
                                                    norm_center_x.unsqueeze(1),
                                                    norm_center_y.unsqueeze(1),
                                                    norm_bndbox_w.unsqueeze(1),
                                                    norm_bndbox_h.unsqueeze(1)],
                                                   dim=1)

        # 이미지를 텐서로 변환 및 정규화
        img_tensor = fT.to_tensor(img)
        if self.normalize:
            mask_x, mask_y = mask
            fT.normalize(img_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]],
                         mean=self.normalize[0],
                         std=self.normalize[1],
                         inplace=True)

        return img_tensor, target


class ImgToTensor:
    """
    PIL 이미지를 텐서로 변환하는 클래스.
    선택적으로 채널별 정규화를 적용합니다. (사전학습 시 사용)
    """

    def __init__(self, normalize: Optional[List] = None) -> None:
        """
        :param normalize: [평균값 리스트, 표준편차 리스트] (채널별 정규화용, None이면 정규화 안 함)
        """
        self.normalize = normalize

    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[th.Tensor, th.Tensor]:
        """
        이미지를 텐서로 변환하고 선택적으로 정규화합니다. 타겟은 변경하지 않습니다.

        :param sample: (이미지, 마스크, 타겟) 튜플
        :return: (이미지 텐서, 타겟)
        """
        img, mask, target = sample

        img_tensor = fT.to_tensor(img)
        if self.normalize:
            mask_x, mask_y = mask
            img_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]] = fT.normalize(img_tensor,
                                                                                   mean=self.normalize[0],
                                                                                   std=self.normalize[1])
        return img_tensor, target
