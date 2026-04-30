## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = './datasets'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nframe = img_label.n_frames

## 데이터셋 분할 설정 (Train/Val/Test 비율)
nframe_train = 24  # 학습용 데이터 프레임 수
nframe_val = 3     # 검증용 데이터 프레임 수
nframe_test = 3    # 테스트용 데이터 프레임 수

dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

##
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

## 학습 데이터 저장
offset_nframe = 0 # 시작 위치 초기화

for i in range(nframe_train):  # 학습용 프레임 수 만큼 반복
    img_label.seek(id_frame[i + offset_nframe])  # 해당 프레임 인덱스에 해당하는 라벨 이미지 선택
    img_input.seek(id_frame[i + offset_nframe])  # 해당 프레임 인덱스에 해당하는 입력 이미지 선택

    label_ = np.asarray(img_label)  # 라벨 이미지를 numpy 배열로 변환
    input_ = np.asarray(img_input)  # 입력 이미지를 numpy 배열로 변환

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)  
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)  
    # 각각의 라벨 및 입력 이미지를 npy 파일로 저장 (파일명은 3자리 숫자)

## 검증 데이터 저장
offset_nframe = nframe_train # 학습 프레임 이후 위치부터 시작

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])  # 검증 프레임 인덱스의 라벨 선택
    img_input.seek(id_frame[i + offset_nframe])  # 검증 프레임 인덱스의 입력 선택

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

## 테스트 데이터 저장
offset_nframe = nframe_train + nframe_val # 학습 + 검증 이후 프레임부터 시작

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

## 시각화
plt.subplot(121)
plt.imshow(label_, cmap='gray') 
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()








