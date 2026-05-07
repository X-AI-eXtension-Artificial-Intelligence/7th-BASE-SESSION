# UNET 구현하기

## Model Architecture 공부

### data_read.py
- 데이터 불러오기
- 데이터 전처리(행렬 변환)

### dataset.py
- 데이터 로더 구현
- 트랜스폼 구현
- 정규화 구현
- 데이터 증강(회전) 구현

### display_results.py
- Input / Label / Output 불러오기
- 각각 시각화

### eval.py
- 모델 평가 구현

### model.py
- 모델 아키텍처 구현

### run_unet.ipynb
- 모델 학습 실행

### train.py
- UNET 네트워크 생성
- UNET 학습 로직 구현

### util.py
- 네트워크 저장 구현
- 네트워크 로더 구현