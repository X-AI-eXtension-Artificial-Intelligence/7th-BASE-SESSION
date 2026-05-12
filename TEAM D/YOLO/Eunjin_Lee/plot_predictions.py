"""
plot_predictions.py - YOLOv1 예측 결과 시각화

학습된 YOLOv1 모델의 바운딩 박스 예측을 인터랙티브하게 시각화합니다.

조작법:
  - ← / → 화살표: 이전/다음 이미지로 이동
  - S: 현재 어노테이션된 이미지 저장
  - Q: 프로그램 종료
"""

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

# ===== 모델 하이퍼파라미터 =====
S = 7       # 그리드 크기
B = 2       # 셀당 바운딩 박스 수
D = 448     # 입력 이미지 크기

# ===== 경로 설정 =====
TRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                        "Detection/checkpoints/trained_model_weights.pt"
PASCAL_VOC_DIR_PATH = "/media/soul/DATA/cv_datasets/PASCAL_VOC/VOC_Detection"
ASSETS_DIR = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object Detection/assets"

# ===== 디바이스 설정 =====
DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

# ===== 후처리 하이퍼파라미터 =====
PROB_THRESHOLD = 0.15   # 시각화용 신뢰도 임계값 (mAP 계산보다 높게 설정)
NMS_THESHOLD = 0.6      # NMS IoU 임계값
CONF_MODE = 'objectness'  # 신뢰도 모드: objectness만 사용


# 전역 변수: 키보드 이벤트 처리용
global mv, save


def on_key_press(event: matplotlib.backend_bases.Event) -> None:
    """
    키보드 이벤트 핸들러.
      - ← : 이전 이미지 (mv = -1)
      - → : 다음 이미지 (mv = +1)
      - S  : 현재 이미지 저장 (save = True)
      - Q  : 프로그램 종료

    :param event: matplotlib 키보드 이벤트
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


def annotate_img(img: Image.Image,
                 bboxes: th.Tensor
                 ) -> Image:
    """
    이미지에 바운딩 박스와 클래스 레이블을 그려 어노테이션합니다.

    :param img: 원본 PIL 이미지
    :param bboxes: 탐지 결과 (N, 6): [클래스, 신뢰도, xmin, ymin, xmax, ymax]
    :return: 바운딩 박스가 그려진 PIL 이미지
    """
    img_tensor = fT.pil_to_tensor(img)
    bboxes_coords = bboxes[:, 2:]

    bboxes_class = bboxes[:, 0].long()
    objectness = bboxes[:, 1]
    # 레이블 텍스트: "클래스명: 신뢰도%"
    text = [f'{VOC_Detection.index2label[bb_class_ind]}: {objectness[i] * 100:.1f}%' for i, bb_class_ind in
            enumerate(bboxes_class)]
    # 클래스별 색상 지정
    obj_clrs = [VOC_Detection.label_clrs[bb_class_ind] for bb_class_ind in bboxes_class]

    annotated_tensor = draw_bounding_boxes(img_tensor, bboxes_coords, text, width=4, font_size=20, colors=obj_clrs)
    annotated_img = fT.to_pil_image(annotated_tensor)
    return annotated_img


def update_plot(ax: matplotlib.axes.Axes,
                model: YOLOv1,
                img: Image.Image
                ) -> Image.Image:
    """
    이미지에 대해 모델 예측을 수행하고 결과를 시각화합니다.

    처리 과정:
    1. 이미지를 D×D로 리사이즈 + 정규화
    2. 모델 추론
    3. 후처리 (좌표 역정규화 + NMS)
    4. 원본 이미지 크기로 좌표 스케일링
    5. 바운딩 박스 그리기

    :param ax: matplotlib 축 객체
    :param model: 학습된 YOLOv1 모델
    :param img: 원본 크기의 PIL 이미지
    :return: 어노테이션된 이미지
    """
    w, h = img.size
    # 전처리: 리사이즈 → 텐서 → 정규화
    x = fT.normalize(fT.to_tensor(fT.resize(img, (D, D))),
                     mean=[0.4549, 0.4341, 0.4010],
                     std=[0.2703, 0.2672, 0.2808]).unsqueeze(0).to(DEVICE)

    # 모델 추론 + 후처리
    with th.no_grad():
        y = model(x)
    bboxes_pred = postprocessing(y,
                                 prob_threshold=PROB_THRESHOLD,
                                 conf_mode=CONF_MODE,
                                 nms_threshold=NMS_THESHOLD)

    # 후처리 좌표는 D×D 기준 → 원본 이미지 크기로 스케일링
    bboxes_pred[:, [2, 4]] *= w / D  # x 좌표
    bboxes_pred[:, [3, 5]] *= h / D  # y 좌표

    # 어노테이션 및 표시
    img = annotate_img(img, bboxes_pred)
    ax.imshow(img)
    plt.show()

    return img


def setup_evaluation() -> Tuple[YOLOv1, VOC_Detection]:
    """
    시각화에 필요한 모델과 데이터셋을 초기화합니다.
    모델은 평가 모드로 설정됩니다.

    :return: (학습된 YOLOv1 모델, PASCAL VOC 테스트 데이터셋)
    """
    model = YOLOv1(S=S,
                   B=B,
                   C=VOC_Detection.C).to(DEVICE)
    trained_model_weights = th.load(TRAINED_MODEL_WEIGHTS)
    model.load_state_dict(trained_model_weights)
    model.eval()

    # 변환 없이 원본 이미지 로드 (시각화용)
    test_dataset = VOC_Detection(root_dir=PASCAL_VOC_DIR_PATH,
                                 split='test')

    return model, test_dataset


def main():
    """
    인터랙티브 시각화 메인 루프.
    matplotlib의 interactive 모드를 사용하여 키보드 입력으로 이미지를 탐색합니다.
    """
    model, test_dataset = setup_evaluation()

    global mv, save
    plt.ion()  # interactive 모드 활성화
    fig, ax = plt.subplots()
    ax.axis('off')
    # 기본 키 핸들러 비활성화 후 커스텀 핸들러 등록
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # 첫 번째 이미지 표시
    i = 0
    img = test_dataset[i][0]
    annot_img = update_plot(ax, model, img)
    mv = 0
    save = False

    # 이벤트 루프
    while True:
        if mv != 0:
            # 이전/다음 이미지로 이동 (순환)
            i = (i + mv) % len(test_dataset)
            img = test_dataset[i][0]
            annot_img = update_plot(ax, model, img)
        if save:
            # 현재 어노테이션 이미지 저장
            path = os.path.join(ASSETS_DIR, f'annnot_img_{i}.jpg')
            annot_img.save(path)
        mv = 0
        save = False
        plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
