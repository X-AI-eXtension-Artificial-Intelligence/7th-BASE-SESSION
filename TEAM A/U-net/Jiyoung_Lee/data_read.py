"""
데이터 전처리 스크립트

역할:
- tif 형태의 원본 데이터를 읽어서
- train / val / test로 분할
- numpy(.npy) 형태로 저장하여 학습 속도 향상
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

dir_data = './datasets'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

# tif 파일 로드 (multi-frame)
img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

nframe = img_label.n_frames

# =========================
# 데이터 분할
# =========================
nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

os.makedirs(dir_save_train, exist_ok=True)
os.makedirs(dir_save_val, exist_ok=True)
os.makedirs(dir_save_test, exist_ok=True)

# 랜덤 셔플
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# =========================
# train 저장
# =========================
offset = 0
for i in range(nframe_train):
    img_label.seek(id_frame[i + offset])
    img_input.seek(id_frame[i + offset])

    np.save(os.path.join(dir_save_train, f'label_{i:03d}.npy'), np.asarray(img_label))
    np.save(os.path.join(dir_save_train, f'input_{i:03d}.npy'), np.asarray(img_input))

# =========================
# val 저장
# =========================
offset = nframe_train
for i in range(nframe_val):
    img_label.seek(id_frame[i + offset])
    img_input.seek(id_frame[i + offset])

    np.save(os.path.join(dir_save_val, f'label_{i:03d}.npy'), np.asarray(img_label))
    np.save(os.path.join(dir_save_val, f'input_{i:03d}.npy'), np.asarray(img_input))

# =========================
# test 저장
# =========================
offset = nframe_train + nframe_val
for i in range(nframe_test):
    img_label.seek(id_frame[i + offset])
    img_input.seek(id_frame[i + offset])

    np.save(os.path.join(dir_save_test, f'label_{i:03d}.npy'), np.asarray(img_label))
    np.save(os.path.join(dir_save_test, f'input_{i:03d}.npy'), np.asarray(img_input))

# 시각화
plt.subplot(121)
plt.imshow(np.asarray(img_label), cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(np.asarray(img_input), cmap='gray')
plt.title('input')

plt.show()