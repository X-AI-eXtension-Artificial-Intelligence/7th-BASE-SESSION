## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = './datasets'

name_label = 'train-labels.tif' # 정답지 (Ground Truth)
name_input = 'train-volume.tif' # 입력 영상 (Raw Data)
# TIFF 파일 오픈 (Multi-page 지원)
img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))
# 이미지의 가로, 세로 크기 및 전체 프레임(장수) 파악
ny, nx = img_label.size
nframe = img_label.n_frames

# 데이터 개수 설정(30개라고 가정함)
nframe_train = 24
nframe_val = 3
nframe_test = 3
# 저장할 디렉토리 경로 설정
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')
# 폴더가 없으면 자동으로 생성하는 안전장치
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

# 인덱스 생성 및 무작위 섞기(데이터의 순서가 편향되지 않도록)
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# 학습 데이터 저장 (Train)
offset_nframe = 0
    
for i in range(nframe_train):
    # 섞인 인덱스에서 i번째 프레임을 선택하여 이동
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])
    # PIL 이미지를 넘파이 배열로 변환
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

# 검증 데이터 저장 (Validation) - 학습 데이터 이후부터 시작
offset_nframe = nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

# 테스트 데이터 저장 (Test) - 학습+검증 데이터 이후부터 시작
offset_nframe = nframe_train + nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

##
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()








