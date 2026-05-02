import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
# 시스템 기준 절대 경로 설정
base_dir= os.path.dirname(os.path.abspath(__file__))
#데이터 파일 이름
train_label="train-labels.tif"
train_input="train-volume.tif"
test_input="test-volume.tif"

#이미지 저장
img_label=Image.open(os.path.join(base_dir,"datasets",train_label))
img_input=Image.open(os.path.join(base_dir,"datasets",train_input))

print(img_label.size)

dir_save_train=os.path.join(base_dir,"datasets","train")
dir_save_val=os.path.join(base_dir,"datasets","val")
dir_save_test=os.path.join(base_dir,"datasets","test")

if not os.path.exists(dir_save_train):
    os.mkdir(dir_save_train)
if not os.path.exists(dir_save_val):
    os.mkdir(dir_save_val)
if not os.path.exists(dir_save_test):
    os.mkdir(dir_save_test)

offset_nframe=0
nframe=img_input.size
nframe_train=24
nframe_val=3
nframe_test=3
id_nframe=np.arange(nframe)
np.random.suffle(id_nframe)


for i in range(nframe_train):
    img_label.seek(id_nframe[offset_nframe+i])
    img_input.seek(id_nframe[offset_nframe+i])

    label_=np.asarry(img_label)
    input_=np.asarry(img_input)

    np.save(os.path.join(dir_save_train,'label_%03d.npy'%i),label_)
    np.save(os.path.join(dir_save_train,'input_%03d.npy'%i),input_)

#val
offset_nframe+=nframe_train


for i in range(nframe_val):
    img_label.seek(id_nframe[offset_nframe+i])
    img_input.seek(id_nframe[offset_nframe+i])

    label_=np.asarry(img_label)
    input_=np.asarry(img_input)

    np.save(os.path.join(dir_save_val,'label_%03d.npy'%i),label_)
    np.save(os.path.join(dir_save_val,'input_%03d.npy'%i),input_)

#test
offset_nframe+=nframe_val


for i in range(nframe_test):
    img_label.seek(id_nframe[offset_nframe+i])
    img_input.seek(id_nframe[offset_nframe+i])

    label_=np.asarry(img_label)
    input_=np.asarry(img_input)

    np.save(os.path.join(dir_save_test,'label_%03d.npy'%i),label_)
    np.save(os.path.join(dir_save_test,'input_%03d.npy'%i),input_)