import os
import numpy as np
import matplotlib.pyplot as plt

# 결과를 시각화하는 코드(예측값)
result_dir = './results/numpy'

lst_data = os.listdir(result_dir)
# 파일 이름의 시작 부분에 따라 input, label, output(모델의 예측값)을 구분
lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]
# 파일 순서가 꼬이지 않도록 정렬 (매우 중요!)
lst_label.sort()
lst_input.sort()
lst_output.sort()

## 특정 데이터 로드
id = 0 # 확인하고 싶은 데이터의 번호

label = np.load(os.path.join(result_dir, lst_label[id])) # 정답 (Ground Truth)
input = np.load(os.path.join(result_dir, lst_input[id])) # 모델에 들어갔던 원본 영상
output = np.load(os.path.join(result_dir, lst_output[id])) # 모델이 뱉어낸 예측 영상

## 세 장의 이미지를 나란히 놓아 모델이 얼마나 정답을 잘 맞췄는지 비교
plt.subplot(131) # 1행 3열 중 첫 번째
plt.imshow(input, cmap='gray')
plt.title('input')

plt.subplot(132) # 1행 3열 중 두 번째
plt.imshow(label, cmap='gray')
plt.title('label')

plt.subplot(133) # 1행 3열 중 세 번째
plt.imshow(output, cmap='gray')
plt.title('output')

plt.show()








