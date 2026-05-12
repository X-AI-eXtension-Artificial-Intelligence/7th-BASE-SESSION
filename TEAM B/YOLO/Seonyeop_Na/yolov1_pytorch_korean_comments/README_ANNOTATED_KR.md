# yolov1_pytorch 한국어 주석 가이드

이 자료는 nsoul97/yolov1_pytorch 저장소를 처음 읽는 사람이 흐름을 이해하기 쉽도록 정리한 학습용 주석 버전입니다.
원본 전체 코드를 그대로 복제한 파일이 아니라, 핵심 구조와 함수 역할을 한국어 주석 중심으로 재구성했습니다.

## 전체 실행 흐름

1. `dataset.py`
   - PASCAL VOC 이미지와 CSV 형태의 박스 라벨을 읽습니다.
   - 각 샘플은 `(이미지, 정답 박스들)` 형태로 반환됩니다.

2. `transforms.py`
   - 이미지를 448x448로 맞춥니다.
   - 확대, 축소, 좌우반전, 색상 변형 같은 데이터 증강을 수행합니다.
   - 마지막에 정답 박스를 YOLO 형식인 `(S, S, C+5)` 텐서로 바꿉니다.

3. `model.py`
   - YOLOv1 네트워크 구조를 정의합니다.
   - ImageNet 사전학습용 classification mode와 VOC 객체 탐지용 detection mode를 모두 지원합니다.
   - detection mode에서는 출력 shape가 `(batch, S, S, C + B*5)`가 됩니다.

4. `loss.py`
   - YOLO 논문의 손실함수를 구현합니다.
   - 위치 손실, objectness 손실, classification 손실을 합칩니다.
   - 한 셀에서 여러 박스가 나오면 IoU가 가장 높은 박스를 책임 박스로 선택합니다.

5. `train.py`
   - 모델, 데이터셋, optimizer, scheduler, loss를 만들고 학습을 반복합니다.
   - `SUBDIVISIONS`를 사용해 gradient accumulation을 수행합니다.

6. `evaluate.py`
   - 모델 예측값을 실제 박스 좌표로 복원합니다.
   - confidence threshold와 NMS를 적용합니다.
   - PASCAL VOC 방식의 mAP를 계산합니다.

7. `plot_predictions.py`
   - 학습된 모델로 예측한 박스를 이미지 위에 그려 시각화합니다.

## YOLOv1 출력 구조

YOLOv1은 이미지를 `S x S` 격자로 나눕니다.
각 격자 셀은 다음 값을 예측합니다.

- 클래스 확률 `C`개
- bounding box `B`개
- 각 bounding box는 `confidence, x, y, w, h` 총 5개 값

따라서 한 셀의 출력 길이는 다음과 같습니다.

```python
C + B * 5
```

보통 이 코드에서는 다음 설정을 사용합니다.

```python
S = 7      # 7x7 grid
B = 2      # 셀마다 박스 2개
C = 20     # PASCAL VOC 클래스 수
D = 448    # 입력 이미지 크기
```

한 이미지의 최종 출력 shape는 다음과 같습니다.

```python
(7, 7, 20 + 2 * 5) = (7, 7, 30)
```
