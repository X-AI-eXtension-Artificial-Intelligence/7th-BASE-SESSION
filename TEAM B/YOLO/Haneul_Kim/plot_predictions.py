"""
학습된 YOLOv1 모델의 예측 결과를 이미지 위에 시각화하는 파일입니다.
테스트 이미지에 대해 bounding box와 class label을 그려서 확인할 수 있습니다.
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

# Model Hyperparameters
# YOLOv1은 이미지를 S x S grid로 나눕니
S = 7
# 각 grid cell마다 예측하는 bounding box 개수
B = 2
# 모델 입력 이미지 크기 즉 448 x 448로 resize
D = 448

# Trained Model Path
TRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                        "Detection/checkpoints/trained_model_weights.pt"

# VOC Dataset Directory
PASCAL_VOC_DIR_PATH = "/media/soul/DATA/cv_datasets/PASCAL_VOC/VOC_Detection"

# Save Image Path
ASSETS_DIR = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object Detection/assets"

# Compute Device (use a GPU if available)
# GPU가 있으면 CUDA를 사용하고, 없으면 CPU를 사용
DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

# Postprocessing Hyperparameters
PROB_THRESHOLD = 0.15
NMS_THESHOLD = 0.6
CONF_MODE = 'objectness'


# 키보드 입력에 따라 이미지 이동/저장 여부를 제어하는 전역 변수
global mv, save


# 키보드 입력을 받아 이미지 이동, 저장, 종료 동작을 처리
def on_key_press(event: matplotlib.backend_bases.Event) -> None:
    """
    Set the mv global variable as +1 or -1 when the arrow keys are pressed. The mv flag is used to change the current
    image of the test set. When the Q is pressed, the program exits. When the key S is pressed, the global variable
    save is updated to True to save the current image. Otherwise, the global variables are updated for nothing to
    happen.

    :param event: An event that is triggered when a key is pressed
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


# 예측된 bounding box와 label을 이미지 위에 그리는 함수
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

 # PIL 이미지를 bounding box drawing이 가능한 tensor로 변환
    img_tensor = fT.pil_to_tensor(img)
    bboxes_coords = bboxes[:, 2:]

    bboxes_class = bboxes[:, 0].long()
    objectness = bboxes[:, 1]
 # class 이름과 confidence를 화면에 표시할 문자열로 생성
    text = [f'{VOC_Detection.index2label[bb_class_ind]}: {objectness[i] * 100:.1f}%' for i, bb_class_ind in
            enumerate(bboxes_class)]
    obj_clrs = [VOC_Detection.label_clrs[bb_class_ind] for bb_class_ind in bboxes_class]

    annotated_tensor = draw_bounding_boxes(img_tensor, bboxes_coords, text, width=4, font_size=20, colors=obj_clrs)
    annotated_img = fT.to_pil_image(annotated_tensor)
    return annotated_img


# 한 장의 이미지에 대해 모델 예측을 수행하고 결과를 시각화
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
 # 원본 이미지의 width, height를 가져오기
        w, h = img.size
    x = fT.normalize(fT.to_tensor(fT.resize(img, (D, D))),
                     mean=[0.4549, 0.4341, 0.4010],
                     std=[0.2703, 0.2672, 0.2808]).unsqueeze(0).to(DEVICE)

 # 시각화/평가 단계에서는 gradient 계산이 필요 없으므로 비활성화
    with th.no_grad():
        y = model(x)
    bboxes_pred = postprocessing(y,
                                 prob_threshold=PROB_THRESHOLD,
                                 conf_mode=CONF_MODE,
                                 nms_threshold=NMS_THESHOLD)

 # After postprocessing, the bounding box coordinates are scaled for a (D x D) image.
    bboxes_pred[:, [2, 4]] *= w / D
    bboxes_pred[:, [3, 5]] *= h / D

    img = annotate_img(img, bboxes_pred)
    ax.imshow(img)
    plt.show()

    return img


# 모델과 테스트 데이터셋을 준비하는 함수
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
 # BatchNorm/Dropout 등을 평가 모드로 전환
    model.eval()

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
    plt.ion()
    fig, ax = plt.subplots()
    ax.axis('off')
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    i = 0
    img = test_dataset[i][0]
    annot_img = update_plot(ax, model, img)
    mv = 0
    save = False
    while True:
        if mv != 0:
            i = (i + mv) % len(test_dataset)
            img = test_dataset[i][0]
            annot_img = update_plot(ax, model, img)
        if save:
            path = os.path.join(ASSETS_DIR, f'annnot_img_{i}.jpg')
            annot_img.save(path)
        mv = 0
        save = False
        plt.waitforbuttonpress()


if __name__ == '__main__':
    main()