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

# (중략 - 하이퍼파라미터 및 경로 설정)
PROB_THRESHOLD = 0.15 # 15% 이상 확신하는 박스만 그림
NMS_THESHOLD = 0.6    # 겹치는 박스 제거 기준

global mv, save

# --- 키보드 인터랙션 이벤트 ---
def on_key_press(event: matplotlib.backend_bases.Event) -> None:
    """왼쪽/오른쪽 화살표 키로 이미지를 넘기고, 's'를 눌러 저장, 'q'를 눌러 종료하는 인터페이스"""
    global mv, save
    if event.key == 'left':
        mv = -1   # 이전 이미지
    elif event.key == 'right':
        mv = +1   # 다음 이미지
    elif event.key == 's':
        save = True # 이미지 저장
    # ...

# --- 이미지 위에 박스 그리기 ---
def annotate_img(img: Image.Image, bboxes: th.Tensor) -> Image:
    """모델이 예측한 박스 좌표와 클래스 이름, 확률을 이미지 텐서 위에 오버레이"""
    img_tensor = fT.pil_to_tensor(img)
    bboxes_coords = bboxes[:, 2:] # [xmin, ymin, xmax, ymax] 부분만 추출
    bboxes_class = bboxes[:, 0].long()
    
    # 박스 위에 띄울 텍스트 생성 (예: "dog: 85.2%")
    text = [f'{VOC_Detection.index2label[bb_class_ind]}: {objectness[i] * 100:.1f}%' ...]
    obj_clrs = [VOC_Detection.label_clrs[bb_class_ind] for bb_class_ind in bboxes_class]

    # torchvision 제공 함수를 사용해 박스 그리기
    annotated_tensor = draw_bounding_boxes(img_tensor, bboxes_coords, text, width=4, font_size=20, colors=obj_clrs)
    annotated_img = fT.to_pil_image(annotated_tensor)
    return annotated_img

# --- 예측 및 화면 업데이트 ---
def update_plot(ax: matplotlib.axes.Axes, model: YOLOv1, img: Image.Image) -> Image.Image:
    """이미지를 모델에 넣고 추론한 뒤, 후처리를 거쳐 화면을 갱신합니다."""
    # 이미지를 모델 입력 크기(D x D)로 리사이징하고 정규화
    x = fT.normalize(fT.to_tensor(fT.resize(img, (D, D))), ...).unsqueeze(0).to(DEVICE)

    with th.no_grad():
        y = model(x) # 모델 추론 (Forward)
        
    # NMS 적용, IOU 낮은 거 버리기 등 후처리 진행
    bboxes_pred = postprocessing(y, prob_threshold=PROB_THRESHOLD,
                                 conf_mode=CONF_MODE, nms_threshold=NMS_THESHOLD)

    # 출력된 (D x D) 기준 좌표를 원래 이미지의 원본 해상도 비율(w, h)에 맞게 복구
    bboxes_pred[:, [2, 4]] *= w / D
    bboxes_pred[:, [3, 5]] *= h / D

    img = annotate_img(img, bboxes_pred) # 이미지에 박스 그리기 적용
    ax.imshow(img)                       # matplotlib 화면 업데이트
    plt.show()

    return img

def setup_evaluation() -> Tuple[YOLOv1, VOC_Detection]:
    # 학습 완료된 모델의 가중치(Trained Weights)를 로드하여 평가 모드(eval)로 세팅
    # ...

def main():
    """Matplotlib을 띄워두고 키보드 이벤트 무한 루프를 돌며 대화형 뷰어 실행"""
    model, test_dataset = setup_evaluation()
    # matplotlib 창 설정 및 이벤트 핸들러 연결
    # 화살표 입력에 따라 `test_dataset`에서 이미지를 꺼내와 `update_plot` 반복 호출
    # ...

if __name__ == '__main__':
    main()