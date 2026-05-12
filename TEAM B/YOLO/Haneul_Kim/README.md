# YOLOv1 코드 주석본

이 폴더는 업로드된 YOLOv1 코드에 한국어 주석을 추가한 버전입니다.

## 파일 구성

- `dataset.py`: PASCAL VOC 이미지와 annotation을 불러오는 Dataset 클래스
- `model.py`: YOLOv1 모델 구조, convolution module, locally connected layer
- `loss.py`: YOLOv1 loss, bounding box corner 변환, IoU 계산
- `train.py`: PASCAL VOC detection 학습 코드
- `pretrain.py`: ImageNet classification 사전학습 코드
- `evaluate.py`: 후처리, NMS, mAP 평가 코드
- `plot_predictions.py`: 예측 결과 시각화 코드
- `transforms.py`: 이미지 증강 및 YOLO target 변환 코드

원본 코드의 동작을 바꾸지 않기 위해, 주로 설명용 주석만 추가했습니다.
