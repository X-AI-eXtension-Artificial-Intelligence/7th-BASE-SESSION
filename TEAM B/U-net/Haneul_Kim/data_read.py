## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = './datasets'

name_label = 'train-labels.tif' # 정답
name_input = 'train-volume.tif' # 입력 이미지

img_label = Image.open(os.path.join(dir_data, name_label)) # tif 파일 열어 객체로 저장
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size # 이미지 크기
nframe = img_label.n_frames # tif 안에 들어있는 이미지 개수

## 데이터 셋 분할
nframe_train = 24 # train 개수
nframe_val = 3 # val 개수
nframe_test = 3 # test 개수

dir_save_train = os.path.join(dir_data, 'train') # 저장 폴더 생성
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train): # 없으면 새로 만들기
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

## 데이터 섞기 (데이터 편향 방지)
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

## train 데이터 저장
offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe]) # tif에서 특정 frame 선택, 랜덤 인덱스
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label) # PIL 이미지를 nmpy 배열로 변환
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_) # .npy로 저장
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

## validation 데이터 만드는 코드
offset_nframe = nframe_train # train으로 사용한 데이터

for i in range(nframe_val): # train 다음부터 val로 사용
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

## test 데이터 만드는 코드
offset_nframe = nframe_train + nframe_val # train + val에서 이미 사용한 데이터

for i in range(nframe_test): # train + val 다음부터 test로 사용
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

## 데이터 로드 확인 (시각화)
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()








