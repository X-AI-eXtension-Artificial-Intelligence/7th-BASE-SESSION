import torch as th
from torch.nn.functional import one_hot
import torchvision.transforms.functional as fT
from PIL.Image import Image
from typing import Tuple, Optional, List


# ============================================================
# [transforms.py 개요]
# 이 파일은 YOLO 학습/평가 파이프라인에서 사용되는 모든 이미지 변환
# 클래스를 정의합니다. 각 클래스는 __call__을 구현하여 callable로
# 동작하며, torchvision.transforms.Compose와 함께 체이닝됩니다.
#
# 전체 변환 흐름 (train 기준):
#   RandomScaleTranslate → RandomColorJitter → RandomHorizontalFlip
#   → ToYOLOTensor
#
# 전체 변환 흐름 (evaluation 기준):
#   Resize → ImgToTensor
#
# [mask란?]
# zoom out 시 이미지 주변이 zero-padding되는데, 이 패딩 영역에
# color jitter나 normalization을 적용하면 의미 없는 값이 왜곡됩니다.
# 이를 방지하기 위해 실제 이미지가 존재하는 영역을 mask로 추적하고,
# 이후 변환에서 mask 영역에만 연산을 적용합니다.
# mask = [(x_start, x_end), (y_start, y_end)]
# ============================================================


# ============================================================
# [Resize]
# evaluation 파이프라인에서 사용되는 단순 리사이즈 변환입니다.
# augmentation 없이 이미지를 (d x d)로 리사이즈하고,
# bounding box 좌표를 비율에 맞게 스케일링합니다.
# ============================================================
class Resize:
    """
    A callable Resize class, which upon its call resizes the image and scales the bounding box coordinates
    appropriately.
    """

    def __init__(self, output_size: int) -> None:
        """
        Initialize the dimension d of the image after the transformation. After the image is resized, it will have a
        (d x d) shape.

        :param output_size: The dimension of the image after the transformation.
        """
        self.d = output_size

    def __call__(self, sample: Tuple[Image, th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        Resize the image to a (d x d) shape and transform the bounding box coordinates.
        In an image with N objects, the target tensor has a (N x 5)-shape and for each object the target is formatted
        as  <classification_id>, <x_min>, <y_min>, <x_max>, <y_max>. Given an (h x w)-image, the x and y coordinates
        are updated to x' and y' in the following way:

        | x' = x * d / w
        | y' = y * d / h

        :param sample: A tuple containing the image and its corresponding target
        :return: The resized (d x d) image, a mask that contains all the image pixels ([0,d] both in the x- and y-axis)
                 and the appropriately scaled coordinates
        """
        img, target = sample
        w, h = img.size

        img = fT.resize(img, (self.d, self.d))
        # x 좌표(xmin, xmax)는 width 비율로, y 좌표(ymin, ymax)는 height 비율로 스케일링합니다.
        target[:, [1, 3]] *= self.d / w
        target[:, [2, 4]] *= self.d / h

        # padding이 없으므로 mask는 이미지 전체 영역을 가리킵니다.
        mask = [(0, self.d), (0, self.d)]
        return img, mask, target


# ============================================================
# [RandomScaleTranslate]
# train 파이프라인의 첫 번째 augmentation입니다.
# 논문 Section 2.2에서 언급된 "random scaling and translations
# of up to 20% of the original image size"를 구현합니다.
#
# 세 가지 operation 중 하나를 확률적으로 선택합니다:
#   - resize    : 단순 리사이즈 (augmentation 없음)
#   - zoom out  : 이미지를 축소 후 zero-padding → 작은 객체 학습에 유리
#   - zoom in   : 이미지 일부를 crop 후 확대 → 큰 객체/부분 객체 학습에 유리
# ============================================================
class RandomScaleTranslate:
    """
    A callable RandomScaleTranslate class, which resizes the image and scales the bounding box coordinates. In order to
    augment the dataset, for each image we randomly choose between the following operations:

    - resize
    - zoom out & resize
    - zoom in & resize

    When we zoom out, the image will be padded with zeros. To avoid distorting these zero values
    (e.g. RandomColorJitter, normalization), a mask is returned to specify which values were padded.
    """
    def __init__(self,
                 output_size: int,
                 jitter: float,
                 resize_p: float,
                 zoom_out_p: float,
                 zoom_in_p: float) -> None:
        """
        Initialize the dimension d of the image after the transformation. After the image is resized, it will have a
        (d x d) shape. The given jitter factor is also stored to randomly scale and translate the image.
        The probabilities are used to select randomly one of the operations.

        :param output_size: The dimension of the image after the transformation.
        :param jitter: A factor to sample the random scale and translation for the zoom operations
        :param resize_p: The probability that the 'resize' operation is applied
        :param zoom_out_p: The probability that the 'zoom out & resize' operation is applied
        :param zoom_in_p: The probability that the 'zoom in & resize' operation is applied
        """
        self.d = output_size
        self.jitter = jitter
        # cumsum으로 누적 확률을 미리 계산해 둡니다.
        # 예: [0.2, 0.4, 0.4] → [0.2, 0.6, 1.0]
        # __call__에서 rand 값과 비교해 operation을 선택합니다.
        self.t_probs = th.cumsum(th.Tensor([resize_p, zoom_out_p, zoom_in_p]), dim=0)

    def __call__(self, sample: Tuple[Image, th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        Sample from a uniform random distribution whether to apply the 'resize', 'zoom out & resize' or 'zoom in &
        resize' operation. The probability of each operation is equal to the given corresponding value.

        In each case, the image is resized to a (d x d) shape and the bounding box coordinates are transformed
        appropriately. A mask that specifies the bounds of the non-padded values of the image is also returned.
        For the 'resize' and the 'zoom in & resize' operations, this mask contains all the pixel values of the image.

        If a bounding box is very small after the transformation, it is removed from the targets.

        :param sample: A tuple containing the image and its corresponding target
        :return: A tuple containing the transformed image, its mask and the updated corresponding target
        """
        # [0, 1) 균일 분포에서 샘플링하여 어떤 operation을 적용할지 결정합니다.
        transform_prob = th.rand(1)
        if transform_prob < self.t_probs[0]:                    # resize
            img, mask, target = self._resize(sample)
        elif transform_prob < self.t_probs[1]:                  # zoom out & resize
            img, mask, target = self._zoom_out(sample)
        else:                                                   # zoom in & resize
            img, mask, target = self._zoom_in(sample)

        # zoom in/out 후 너무 작아진 bounding box는 학습에 노이즈가 되므로 제거합니다.
        # threshold = 0.001 * d (예: 448px 기준 약 0.45px)
        bboxes_w = target[:, 3] - target[:, 1]
        bboxes_h = target[:, 4] - target[:, 2]
        threshold = 0.001 * self.d
        valid_bboxes = th.logical_not(th.logical_or(bboxes_w < threshold, bboxes_h < threshold))
        target = target[valid_bboxes]
        return img, mask, target

    def _resize(self, sample: Tuple[Image, th.Tensor]
                ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        This function follows the same logic with the __call__ function of the Resize class.

        Resize the image to a (d x d) shape and transform the bounding box coordinates.
        In an image with N objects, the target tensor has a (N x 5)-shape and for each object the target is formatted
        as  <classification_id>, <x_min>, <y_min>, <x_max>, <y_max>. Given an (h x w)-image, the x and y coordinates
        are updated to x' and y' in the following way:

        | x' = x * d / w
        | y' = y * d / h

        :param sample: A tuple containing the image and its corresponding target
        :return: A tuple containing the resized image, its mask and the updated corresponding target
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
        First a new aspect ratio is set by sampling randomly from a uniform distribution rand_w and rand_h:

        - rand_w ~ U((1-jitter)w, (1+jitter)w)
        - rand_h ~ U((1-jitter)h, (1+jitter)h)

        and setting:
         new_ar = rand_w / rand_h

        We compare rand_w with rand_h and set the large dimension's size equal with d. The size of the other dimension
        is calculated based on the aspect ratio. Therefore, the selected image patch has a size of (d, k) or (k,d) with
        k <= 1.

        Following this resize transformation, the image patch is randomly translated. To translate the image patch, we
        pad the image with zeros:
        - left and right, if the image patch has a width of k
        - top and bottom, if the image patch has a height of k.

        We randomly sample how many pixels are padded on the left or the top of the image from U(0, d-k). We also pad
        the image with zeros on the right or the bottom to have a (d x d) shape.

        The transformations that are applied to the coordinates of the image are:
        1) resize from (w,h) to (d,k) or (k,d)
        2) translate the image by the number of padded values on the left or the top of the image
        Therefore, these transformations will be applied to the bounding box coordinates.

        The mask of the transformed image will contain the bounds of the non-padded values with mask = [mask_x, mask_y]

        :param sample: A tuple containing the image and its corresponding target
        :return: A tuple containing the transformed image after the 'zoom out & resize' operation, its mask and the
                 updated corresponding target
        """
        img, target = sample
        w, h = img.size

        # jitter로 새로운 가상의 aspect ratio를 샘플링합니다.
        # 이를 통해 단순 축소가 아닌 비율 변화가 포함된 zoom out을 구현합니다.
        dw = w * self.jitter
        dh = h * self.jitter
        rand_w = w + th.Tensor(1).uniform_(-dw, dw)
        rand_h = h + th.Tensor(1).uniform_(-dh, dh)
        new_ar = rand_w / rand_h

        # 긴 쪽을 d에 맞추고, 짧은 쪽은 aspect ratio를 유지하며 계산합니다.
        # 결과적으로 이미지는 (d x nw) 또는 (nh x d) 크기로 리사이즈됩니다.
        if new_ar < 1:
            nh = self.d
            nw = int(nh * new_ar + 0.5)
        else:
            nw = self.d
            nh = int(nw / new_ar + 0.5)

        # 리사이즈된 이미지를 (d x d) 캔버스 내 랜덤 위치에 배치합니다.
        # dx, dy는 이미지 왼쪽/위쪽에 추가되는 padding 크기입니다.
        dx = th.randint(low=0, high=self.d - nw + 1, size=(1,)).item()
        dy = th.randint(low=0, high=self.d - nh + 1, size=(1,)).item()

        # 1단계: 이미지를 (nh x nw)로 리사이즈, bounding box도 동일 비율 스케일링
        img = fT.resize(img, (nh, nw))
        target[:, [1, 3]] *= nw / w
        target[:, [2, 4]] *= nh / h

        # 2단계: zero-padding으로 (d x d)를 맞추고, bounding box에 padding offset을 더합니다.
        img = fT.pad(img, padding=[dx, dy, self.d - nw - dx, self.d - nh - dy])
        target[:, [1, 3]] += dx
        target[:, [2, 4]] += dy

        # 실제 이미지 영역만 mask로 기록합니다. (이후 color jitter, normalization에서 활용)
        mask = [(dx, dx + nw), (dy, dy + nh)]
        return img, mask, target

    def _zoom_in(self, sample: Tuple[Image, th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        First we sample the width and height of an image patch, nw and nh respectively, from a uniform random
        distribution:

        - nw ~ U((1-jitter)w, w)
        - nh ~ U((1-jitter)h, h)

        Similarly we sample dx and dy to crop an image patch from the original image.

        - dx ~ U(0, w-nw)
        - dy ~ U(0, h-nh)

        Following that, the selected image patch is resized to a (d x d) shape.

        The bounding box coordinates are transformed in the following way:
        1) the top, left coordinate of the image patch (dx, dy) must be translated to (0,0)
        2) the image patch is resized from a size of (nw, nh) to (d, d)
        The bounding boxes that are not visible after the transformation are completely removed from the targets, while
        the bounding boxes that are only partially visible have their coordinates clamped to be within the image.

        The mask of the transformed image will contain all the pixel values of the image in both the x- and y-axis.

        :param sample: A tuple containing the image and its corresponding target
        :return: A tuple containing the transformed image after the 'zoom in & resize' operation, its mask and the
                 updated corresponding target
        """
        img, target = sample
        w, h = img.size

        # 원본 이미지의 일부 영역(nw x nh)을 crop할 크기를 샘플링합니다.
        nw = int(th.Tensor(1).uniform_((1 - self.jitter) * w, w) + 0.5)
        nh = int(th.Tensor(1).uniform_((1 - self.jitter) * h, h) + 0.5)
        # crop 시작점(dx, dy)을 샘플링합니다.
        dx = int(th.Tensor(1).uniform_(0, w - nw + 1) + 0.5)
        dy = int(th.Tensor(1).uniform_(0, h - nh + 1) + 0.5)

        # crop된 영역을 (d x d)로 리사이즈합니다.
        img = fT.resized_crop(img, top=dy, left=dx, height=nh, width=nw, size=(self.d, self.d))

        # 1단계: crop offset(dx, dy)만큼 좌표를 이동시킵니다.
        target[:, [1, 3]] -= dx
        target[:, [2, 4]] -= dy
        # 2단계: crop 크기(nw, nh)에서 (d, d)로의 리사이즈 비율을 적용합니다.
        target[:, [1, 3]] *= self.d / nw
        target[:, [2, 4]] *= self.d / nh

        # crop 후 완전히 벗어난 bounding box를 제거합니다.
        target = target[th.logical_not(th.logical_or(th.logical_or(target[:, 3] < 0, target[:, 1] > self.d),
                                                     th.logical_or(target[:, 4] < 0, target[:, 2] > self.d)))]

        # 부분적으로만 보이는 bounding box의 좌표를 이미지 경계 안으로 clamp합니다.
        # Update the bounds of the bounding boxes that are only partially visible
        target[:, [1, 2]] = target[:, [1, 2]].clamp(min=0)
        target[:, [3, 4]] = target[:, [3, 4]].clamp(max=self.d)

        # zoom in은 padding이 없으므로 mask는 이미지 전체입니다.
        mask = [(0, self.d), (0, self.d)]
        return img, mask, target


# ============================================================
# [RandomColorJitter]
# 논문 Section 2.2에서 언급된 "randomly adjust the exposure and
# saturation of the image by up to a factor of 1.5 in the HSV
# color space"를 구현합니다. hue도 추가로 조정합니다.
#
# RGB가 아닌 HSV 공간에서 조작하는 이유는 H/S/V 채널이 각각
# 색조/채도/밝기를 독립적으로 제어하기 때문입니다.
#
# zero-padding 영역(mask 바깥)에는 color jitter를 적용하지 않습니다.
# ============================================================
class RandomColorJitter:
    """
    A callable RandomColorJitter class, which when called distorts the colors of the input image. The target values
    remain unchanged.
    """

    def __init__(self, hue: float, sat: float, exp: float):
        """
        Initialize the hue, saturation and exposure parameters.

        :param hue: The hue parameter. The hue value will be sampled uniformly at random from [-hue, hue].
        :param sat: The saturation parameter. The saturation value will be sampled uniformly at random from [1/sat, sat]
        :param exp: The exposure parameter. The exposure parameter will be sampled uniformly at random from [1/exp, exp]
        """
        self.hue = hue
        self.sat = sat
        self.exp = exp

    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        Sample uniformly at random the hue, saturation and exposure values and distort the colors of the input image.
        The hue, saturation and exposure of the image are adjusted in the HSV color space. Specifically:

        HUE
            pixel_H = pixel_H + rand_hue

            if pixel_H > 1, then
                pixel_H = pixel_H - 1
            else if pixel_H < 0, then
                pixel_H = pixel_H + 1

        SATURATION
            pixel_S = min(pixel_S * rand_sat, 1.0)

        EXPOSURE
            pixel_V = min(pixel_V * rand_exp, 1.0)

        :param sample: A tuple containing the image, its mask and the corresponding target
        :return: The distorted image and its (unchanged) target
        """
        # Sample uniformly at random the hue, saturation and exposure values for this image.
        rand_hue = th.Tensor(1).uniform_(-self.hue, self.hue)
        rand_sat = th.Tensor(1).uniform_(1 / self.sat, self.sat)
        rand_exp = th.Tensor(1).uniform_(1 / self.exp, self.exp)

        # Convert the RGB PIL image to an HSV tensor.
        rgb_img, mask, target = sample
        hsv_img = rgb_img.convert('HSV')
        hsv_tensor = fT.to_tensor(hsv_img)

        # mask 영역(실제 이미지 픽셀)에만 color jitter를 적용합니다.
        # zero-padding 영역을 건드리지 않기 위함입니다.
        mask_x, mask_y = mask
        masked_hsv_tensor = hsv_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]]

        # Adjust hue
        # hue는 circular한 값([0,1])이므로 범위를 벗어나면 wrap-around 처리합니다.
        masked_hsv_tensor[0, :, :] += rand_hue
        masked_hsv_tensor[0, :, :] += (1. * (masked_hsv_tensor[0, :, :] < 0) - 1. * \
                                       (masked_hsv_tensor[0, :, :] > 1)) * th.ones_like(masked_hsv_tensor[0, :, :])
        # Adjust saturation
        masked_hsv_tensor[1, :, :] *= rand_sat
        masked_hsv_tensor[1, :, :] = masked_hsv_tensor[1, :, :].clamp(max=1.0)

        # Adjust exposure
        masked_hsv_tensor[2, :, :] *= rand_exp
        masked_hsv_tensor[2, :, :] = masked_hsv_tensor[2, :, :].clamp(max=1.0)

        # Convert the HSV tensor to an RGB PIL image
        hsv_img = fT.to_pil_image(hsv_tensor, mode='HSV')
        rgb_img = hsv_img.convert('RGB')

        return rgb_img, mask, target


# ============================================================
# [RandomHorizontalFlip]
# 확률 p로 이미지를 좌우 반전합니다.
# 반전 시 bounding box의 x 좌표와 mask의 x 범위도 함께 변환합니다.
# ============================================================
class RandomHorizontalFlip:
    """
    A callable RandomHorizontalFlip class. When called, it is randomly chosen whether the image is flipped horizontally.
    When the image is  flipped, the bounding box coordinates and the mask are also transformed appropriately.
    """
    def __init__(self, p: float) -> None:
        """
        Initialize a RandomHorizontalFlip object and set the probability that the image is flipped.

        :param p: The probability that the horizontal flip transformation is applied
        """
        self.p = p

    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        A number in [0,1) is randomly sampled from the uniform distribution U(0,1) to determine if the horizontal flip
        transformation will be applied. The transformation is applied with probability p. If the image is flipped, the
        xmin and xmax coordinates of the bounding boxes are updated. Furthermore, the mask's component in the x-axis
        is also updated similarly.

        :param sample: A tuple containing the image, its mask and the corresponding target
        :return: If the transformation is applied, the horizontally flipped image, the transformed mask and the
                 transformed target is returned. Otherwise, the input sample is returned.
        """
        apply_transform = th.rand(1) < self.p
        if not apply_transform:
            return sample

        img, mask, target = sample
        w = img.size[0]

        # 좌우 반전 시 xmin' = w - xmax, xmax' = w - xmin 으로 변환됩니다.
        # [1, 3] 인덱스가 각각 xmin, xmax이므로 동시에 교환하며 변환합니다.
        target[:, [1, 3]] = w - target[:, [3, 1]]
        img = fT.hflip(img)

        # mask의 x 범위도 동일한 방식으로 반전합니다.
        start_x, end_x = mask[0]
        mask[0] = (w - end_x, w - start_x)

        return img, mask, target


# ============================================================
# [ToYOLOTensor]
# train 파이프라인의 마지막 변환으로, 두 가지 역할을 합니다:
#
# 1. PIL Image → 정규화된 Tensor 변환
# 2. target 형식 변환:
#    (N, 5) 형태의 [class, xmin, ymin, xmax, ymax]를
#    (S, S, C+5) 형태의 YOLO grid 형식으로 변환합니다.
#
# YOLO grid target 형식 (논문 Section 2):
#   - [0]      : objectness (0 or 1)
#   - [1:C+1]  : one-hot class label
#   - [C+1]    : 셀 내 normalized center x (0~1)
#   - [C+2]    : 셀 내 normalized center y (0~1)
#   - [C+3]    : 이미지 기준 normalized width (0~1)
#   - [C+4]    : 이미지 기준 normalized height (0~1)
# ============================================================
class ToYOLOTensor:
    """
    A callable ToYOLOTensor class. When called the targets of the image will be transformed according to the YOLO
    format, while the PIL image will be converted a Tensor. If the mean and the standard deviation of the input image
    channels are provided, the Tensors are normalized.
    """

    def __init__(self, S: int, C: int, normalize: Optional[List] = None) -> None:
        """
        Initialize the number of grid cells per row/column and the number of classes of the dataset.

        :param S: The S parameter of the YOLO algorithm. Each image is split into an (S x S) grid.
        :param C: The number of classes of the dataset.
        :param normalize: A list that contains two lists, one with the 3 mean values of the pixels per channel and
                          another with the corresponding standard deviations per channel.
        """
        self.S = S
        self.C = C
        self.normalize = normalize

    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[th.Tensor, th.Tensor]:
        """
        The PIL image input is converted to a Tensor and the tensor is (optionally normalized).
        In an image with N objects, the input target tensor has a (N x 5)-shape and for each object the target is
        formatted as <classification_id>, <x_min>, <y_min>, <x_max>, <y_max>. The output target tensor has shape
        (S x S x C+5). For each of the (S x S) cells of the grid:

        - index 0: 0 or 1 if an object exists in that cell
        - indices [1,C]: one hot representation of the object in the cell or 0s everywhere
        - index C+1: normalized center x-coordinate.
        - index C+2: normalized center y-coordinate.
        - index C+3: normalized width of the bounding box
        - index C+4: normalized height of the bounding box

        The center coordinates are normalized as offsets in the grid where the upper-left corner in the grid has
        coordinates (0,0) and the bottom-right corner in the grid has coordinates (1,1).

        The height and the width of the bounding boxes are normalized by the image height and width.

        :param sample: A tuple containing the image, its mask and the corresponding target
        :return: The given image and its target in a YOLO-grid format.
        """
        img, mask, target = sample
        w, h = img.size

        # 각 grid cell의 픽셀 단위 크기를 계산합니다.
        cell_w = w / self.S
        cell_h = h / self.S

        # (xmin, ymin, xmax, ymax) → center 좌표 + width/height 로 변환합니다.
        center_x = (target[:, 1] + target[:, 3]) / 2
        center_y = (target[:, 2] + target[:, 4]) / 2
        bndbox_w = target[:, 3] - target[:, 1]
        bndbox_h = target[:, 4] - target[:, 2]

        label = target[:, 0].long()
        # 각 객체의 center가 속하는 grid cell의 (row, col) 인덱스를 계산합니다.
        center_col = th.div(center_x, cell_w, rounding_mode="trunc").long()
        center_row = th.div(center_y, cell_h, rounding_mode="trunc").long()

        # 논문 표현 방식: center 좌표를 해당 셀 내에서의 상대적 offset(0~1)으로 정규화합니다.
        norm_center_x = (center_x % cell_w) / cell_w
        norm_center_y = (center_y % cell_h) / cell_h
        # width/height는 전체 이미지 크기 기준으로 정규화합니다.
        norm_bndbox_w = bndbox_w / w
        norm_bndbox_h = bndbox_h / h

        # (S, S, C+5) 크기의 zero tensor를 생성하고, 각 객체를 해당 grid cell에 기록합니다.
        # 같은 셀에 여러 객체가 있으면 마지막 객체가 덮어씁니다. (YOLO v1의 한계)
        target = th.zeros((self.S, self.S, self.C + 5))
        target[center_row, center_col, :] = th.cat([th.ones((label.shape[0], 1)),
                                                    one_hot(label, self.C),
                                                    norm_center_x.unsqueeze(1),
                                                    norm_center_y.unsqueeze(1),
                                                    norm_bndbox_w.unsqueeze(1),
                                                    norm_bndbox_h.unsqueeze(1)],
                                                   dim=1)

        img_tensor = fT.to_tensor(img)
        # mask 영역(실제 이미지 픽셀)에만 normalization을 적용합니다.
        if self.normalize:
            mask_x, mask_y = mask
            fT.normalize(img_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]],
                         mean=self.normalize[0],
                         std=self.normalize[1],
                         inplace=True)

        return img_tensor, target


# ============================================================
# [ImgToTensor]
# evaluation 파이프라인에서 Resize 다음에 사용됩니다.
# ToYOLOTensor와 달리 target을 YOLO grid 형식으로 변환하지 않고
# 원본 (N, 5) 형태를 유지합니다.
# evaluate.py에서 mAP 계산 시 ground truth로 그대로 사용하기 위함입니다.
# ============================================================
class ImgToTensor:
    """
        A callable ImgToTensor class. When called the PIL image will be converted a Tensor. If the mean and the standard
        deviation of the input image channels are provided, the Tensors are normalized.
        """

    def __init__(self, normalize: Optional[List] = None) -> None:
        """
        Initialize the number of grid cells per row/column and the number of classes of the dataset.

        :param S: The S parameter of the YOLO algorithm. Each image is split into an (S x S) grid.
        :param C: The number of classes of the dataset.
        :param normalize: A list that contains two lists, one with the 3 mean values of the pixels per channel and
                          another with the corresponding standard deviations per channel.
        """
        self.normalize = normalize

    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[th.Tensor, th.Tensor]:
        """
        The PIL image input is converted to a Tensor and the tensor is (optionally normalized). The targets are not
        modified.

        :param sample: A tuple containing the image and its corresponding target
        :return: An image tensor and the corresponding targets.
        """
        img, mask, target = sample

        img_tensor = fT.to_tensor(img)
        if self.normalize:
            mask_x, mask_y = mask
            img_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]] = fT.normalize(img_tensor,
                                                                                   mean=self.normalize[0],
                                                                                   std=self.normalize[1])
        return img_tensor, target