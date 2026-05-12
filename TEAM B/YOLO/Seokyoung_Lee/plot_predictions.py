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

# ============================================================
# [plot_predictions.py 개요]
# 학습된 모델의 bounding box 예측 결과를 인터랙티브하게 시각화합니다.
# evaluate.py가 정량적 평가(mAP)를 담당한다면,
# 이 파일은 정성적 평가(시각 확인)를 위한 도구입니다.
#
# 키보드 조작:
#   ← / → : 이전/다음 이미지로 이동
#   S      : 현재 annotated 이미지를 파일로 저장
#   Q      : 종료
#
# evaluate.py와의 주요 차이점:
#   - DataLoader가 아닌 Dataset을 직접 사용합니다. (transform 없이 PIL Image 그대로)
#   - PROB_THRESHOLD가 0.15로 높습니다. (시각화 목적이므로 탐지 신뢰도를 높임)
#   - CONF_MODE가 'objectness'입니다. (class probability 곱하지 않음)
#   - 예측 좌표를 원본 이미지 크기에 맞게 역스케일링합니다.
# ============================================================

# Model Hyperparameters
S = 7
B = 2
D = 448

# Trained Model Path
TRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                        "Detection/checkpoints/trained_model_weights.pt"

# VOC Dataset Directory
PASCAL_VOC_DIR_PATH = "/media/soul/DATA/cv_datasets/PASCAL_VOC/VOC_Detection"

# Save Image Path
ASSETS_DIR = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object Detection/assets"

# Compute Device (use a GPU if available)
DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

# Postprocessing Hyperparameters
# 시각화 목적이므로 evaluate.py(0.005)보다 높은 threshold를 사용합니다.
PROB_THRESHOLD = 0.15
NMS_THESHOLD = 0.6
CONF_MODE = 'objectness'


# mv: 이미지 이동 방향 (+1: 다음, -1: 이전, 0: 유지)
# save: 현재 이미지 저장 여부
global mv, save


def on_key_press(event: matplotlib.backend_bases.Event) -> None:
    """
    Set the mv global variable as +1 or -1 when the arrow keys are pressed. The mv flag is used to change the current
    image of the test set. When the Q is pressed, the program exits. When the key S is pressed, the global variable
    save is updated to True to save the current image. Otherwise, the global variables are updated for nothing to
    happen.

    :param event: An event that is triggered when a key is pressed
    """
    # matplotlib의 key_press_event 콜백으로 등록되어 키 입력을 처리합니다.
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
    Annotate the given image based on the given bounding boxes.
    The bounding box is plotted for each object of the image and the corresponding label is also written inside the box.

    :param img:  The PIL image to be annotated
    :param bboxes: The ground truth annotation data of the image
    :return: A PIL Image with the bounding boxes and their corresponding labels plotted.
    """

    img_tensor = fT.pil_to_tensor(img)
    bboxes_coords = bboxes[:, 2:]   # [xmin, ymin, xmax, ymax]

    bboxes_class = bboxes[:, 0].long()
    objectness = bboxes[:, 1]
    # dataset.py의 index2label로 클래스 이름을, label_clrs로 색상을 가져옵니다.
    text = [f'{VOC_Detection.index2label[bb_class_ind]}: {objectness[i] * 100:.1f}%' for i, bb_class_ind in
            enumerate(bboxes_class)]
    obj_clrs = [VOC_Detection.label_clrs[bb_class_ind] for bb_class_ind in bboxes_class]

    # torchvision의 draw_bounding_boxes로 박스와 레이블을 이미지에 그립니다.
    annotated_tensor = draw_bounding_boxes(img_tensor, bboxes_coords, text, width=4, font_size=20, colors=obj_clrs)
    annotated_img = fT.to_pil_image(annotated_tensor)
    return annotated_img


def update_plot(ax: matplotlib.axes.Axes,
                model: YOLOv1,
                img: Image.Image
                ) -> Image.Image:
    """
    Predict and plot the bounding boxes for the given image.

    :param ax: The axis where the annotated image will be plotted.
    :param model: The trained YOLOv1 (detection) model
    :param img: The PIL Image of the test set that will be fed to the YOLO model.
    :return: The input image annotated with the bounding box predictions
    """
    w, h = img.size
    # 원본 이미지를 (D x D)로 리사이즈 후 정규화하여 모델 입력을 준비합니다.
    # DataLoader 없이 단일 이미지를 처리하므로 unsqueeze(0)으로 배치 차원을 추가합니다.
    x = fT.normalize(fT.to_tensor(fT.resize(img, (D, D))),
                     mean=[0.4549, 0.4341, 0.4010],
                     std=[0.2703, 0.2672, 0.2808]).unsqueeze(0).to(DEVICE)

    with th.no_grad():
        y = model(x)
    bboxes_pred = postprocessing(y,
                                 prob_threshold=PROB_THRESHOLD,
                                 conf_mode=CONF_MODE,
                                 nms_threshold=NMS_THESHOLD)

    # postprocessing 후 좌표는 (D x D) 기준입니다.
    # 원본 이미지 크기로 역스케일링하여 원본 이미지에 정확히 표시합니다.
    # After postprocessing, the bounding box coordinates are scaled for a (D x D) image.
    bboxes_pred[:, [2, 4]] *= w / D  # x 좌표 역스케일링
    bboxes_pred[:, [3, 5]] *= h / D  # y 좌표 역스케일링

    img = annotate_img(img, bboxes_pred)
    ax.imshow(img)
    plt.show()

    return img


def setup_evaluation() -> Tuple[YOLOv1, VOC_Detection]:
    """
    Instantiate the model and the PASCAL VOC test dataset. The model's weights are loaded from the checkpoint file that
    was updated at the end of the training. The model will be used in the evaluation mode.

    :return: The trained YOLOv1 (detection) model and the PASCAL VOC test dataset.
    """
    model = YOLOv1(S=S,
                   B=B,
                   C=VOC_Detection.C).to(DEVICE)
    trained_model_weights = th.load(TRAINED_MODEL_WEIGHTS)
    model.load_state_dict(trained_model_weights)
    model.eval()

    # transforms 없이 PIL Image 그대로 반환합니다.
    # 전처리는 update_plot 내에서 직접 처리합니다.
    test_dataset = VOC_Detection(root_dir=PASCAL_VOC_DIR_PATH,
                                 split='test')

    return model, test_dataset


def main():
    """
    Plot the bounding box prediction and the images interactively. To navigate in the test set, press the left/right
    arrow keys. To save the annotated image, press S. To terminate the program, press Q.
    """
    model, test_dataset = setup_evaluation()

    global mv, save
    # plt.ion(): interactive mode로 설정해 plot이 블로킹 없이 업데이트됩니다.
    plt.ion()
    fig, ax = plt.subplots()
    ax.axis('off')
    # matplotlib 기본 키 핸들러를 제거하고 커스텀 on_key_press를 등록합니다.
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    i = 0
    img = test_dataset[i][0]
    annot_img = update_plot(ax, model, img)
    mv = 0
    save = False
    while True:
        if mv != 0:
            # % len(test_dataset)으로 인덱스를 순환시킵니다.
            i = (i + mv) % len(test_dataset)
            img = test_dataset[i][0]
            annot_img = update_plot(ax, model, img)
        if save:
            path = os.path.join(ASSETS_DIR, f'annnot_img_{i}.jpg')
            annot_img.save(path)
        mv = 0
        save = False
        # 키 입력이 있을 때까지 대기합니다.
        plt.waitforbuttonpress()


if __name__ == '__main__':
    main()