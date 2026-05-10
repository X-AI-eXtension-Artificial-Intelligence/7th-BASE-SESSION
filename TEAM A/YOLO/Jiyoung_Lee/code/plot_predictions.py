import os.path
import torch as th

import torchvision.transforms.functional as fT
from torchvision.utils import draw_bounding_boxes

from model import YOLOv1
from dataset import VOC_Detection
from evaluate import postprocessing

import PIL.Image as Image

from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib


# =========================================
# Model Hyperparameters
# =========================================

S = 7
B = 2
D = 448


# =========================================
# 학습된 weight 경로
# =========================================

TRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object Detection/checkpoints/trained_model_weights.pt"


# =========================================
# VOC Dataset 경로
# =========================================

PASCAL_VOC_DIR_PATH = "/media/soul/DATA/cv_datasets/PASCAL_VOC/VOC_Detection"


# =========================================
# 저장 경로
# =========================================

ASSETS_DIR = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object Detection/assets"


# =========================================
# Device 설정
# =========================================

DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'


# =========================================
# Postprocessing Hyperparameters
# =========================================

# detection threshold
PROB_THRESHOLD = 0.15

# NMS threshold
NMS_THESHOLD = 0.6

# confidence mode
CONF_MODE = 'objectness'


# 현재 이미지 이동 flag
global mv, save


def on_key_press(event: matplotlib.backend_bases.Event) -> None:
    """
    키 입력 이벤트 처리

    left:
        이전 이미지

    right:
        다음 이미지

    s:
        이미지 저장

    q:
        종료
    """

    global mv, save

    if event.key == 'left':

        mv = -1
        save = False

    elif event.key == 'right':

        mv = +1
        save = False

    elif event.key == 's':

        mv = 0
        save = True

    elif event.key == 'q':

        exit(0)

    else:

        mv = 0
        save = False


def annotate_img(
        img: Image.Image,
        bboxes: th.Tensor
) -> Image:
    """
    bbox와 label을 이미지 위에 그림
    """

    # PIL -> Tensor
    img_tensor = fT.pil_to_tensor(img)

    # bbox 좌표
    bboxes_coords = bboxes[:, 2:]

    # bbox class
    bboxes_class = bboxes[:, 0].long()

    # confidence
    objectness = bboxes[:, 1]

    """
    표시할 text 생성

    ex)
        person: 92.1%
    """

    text = [
        f'{VOC_Detection.index2label[bb_class_ind]}: {objectness[i] * 100:.1f}%'
        for i, bb_class_ind in enumerate(bboxes_class)
    ]

    # class별 색상
    obj_clrs = [
        VOC_Detection.label_clrs[bb_class_ind]
        for bb_class_ind in bboxes_class
    ]

    # bbox 그리기
    annotated_tensor = draw_bounding_boxes(
        img_tensor,
        bboxes_coords,
        text,
        width=4,
        font_size=20,
        colors=obj_clrs
    )

    # Tensor -> PIL
    annotated_img = fT.to_pil_image(annotated_tensor)

    return annotated_img


def update_plot(
        ax: matplotlib.axes.Axes,
        model: YOLOv1,
        img: Image.Image
) -> Image.Image:
    """
    이미지 prediction 수행 후 시각화
    """

    # 원본 이미지 크기
    w, h = img.size

    # =====================================
    # 입력 전처리
    # =====================================

    x = fT.normalize(
        fT.to_tensor(
            fT.resize(img, (D, D))
        ),

        mean=[0.4549, 0.4341, 0.4010],

        std=[0.2703, 0.2672, 0.2808]

    ).unsqueeze(0).to(DEVICE)

    # prediction
    with th.no_grad():

        y = model(x)

    # postprocessing
    bboxes_pred = postprocessing(
        y,
        prob_threshold=PROB_THRESHOLD,
        conf_mode=CONF_MODE,
        nms_threshold=NMS_THESHOLD
    )

    """
    prediction bbox는
    D x D 기준이므로

    원본 이미지 크기로 다시 변환
    """

    bboxes_pred[:, [2, 4]] *= w / D
    bboxes_pred[:, [3, 5]] *= h / D

    # bbox 그리기
    img = annotate_img(img, bboxes_pred)

    # matplotlib 출력
    ax.imshow(img)

    plt.show()

    return img


def setup_evaluation() -> Tuple[YOLOv1, VOC_Detection]:
    """
    model + dataset 생성
    """

    # model 생성
    model = YOLOv1(
        S=S,
        B=B,
        C=VOC_Detection.C
    ).to(DEVICE)

    # trained weight 로드
    trained_model_weights = th.load(
        TRAINED_MODEL_WEIGHTS
    )

    model.load_state_dict(
        trained_model_weights
    )

    # evaluation mode
    model.eval()

    # dataset 생성
    test_dataset = VOC_Detection(
        root_dir=PASCAL_VOC_DIR_PATH,
        split='test'
    )

    return model, test_dataset


def main():
    """
    interactive prediction viewer

    조작:
        left/right:
            이미지 이동

        s:
            저장

        q:
            종료
    """

    # model + dataset 생성
    model, test_dataset = setup_evaluation()

    global mv, save

    # interactive mode
    plt.ion()

    # matplotlib figure 생성
    fig, ax = plt.subplots()

    ax.axis('off')

    # matplotlib 기본 key handler 제거
    fig.canvas.mpl_disconnect(
        fig.canvas.manager.key_press_handler_id
    )

    # 사용자 key event 연결
    fig.canvas.mpl_connect(
        'key_press_event',
        on_key_press
    )

    # 시작 index
    i = 0

    # 첫 이미지
    img = test_dataset[i][0]

    # prediction 출력
    annot_img = update_plot(ax, model, img)

    mv = 0
    save = False

    while True:

        # =====================================
        # 이미지 이동
        # =====================================

        if mv != 0:

            i = (i + mv) % len(test_dataset)

            img = test_dataset[i][0]

            annot_img = update_plot(ax, model, img)

        # =====================================
        # 이미지 저장
        # =====================================

        if save:

            path = os.path.join(
                ASSETS_DIR,
                f'annnot_img_{i}.jpg'
            )

            annot_img.save(path)

        mv = 0
        save = False

        # key 입력 대기
        plt.waitforbuttonpress()


if __name__ == '__main__':
    main()