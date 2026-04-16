## test 결과로 저장된 input, label, output 데이터를 불러와 
## 나란히 시각화하여 모델의 예측 성능을 직관적으로 비교하는 파일

import os
import numpy as np
import matplotlib.pyplot as plt

## label, input, output 파일 분리 & 정렬
result_dir = './results/numpy'

lst_data = os.listdir(result_dir)

lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]

lst_label.sort()
lst_input.sort()
lst_output.sort()

## 동일 인덱스 데이터 불러와 서로 대응 시키기
id = 0

label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

## input, label, output을 나란히 시각화
plt.subplot(131)
plt.imshow(input, cmap='gray')
plt.title('input')

plt.subplot(132)
plt.imshow(label, cmap='gray')
plt.title('label')

plt.subplot(133)
plt.imshow(output, cmap='gray')
plt.title('output')

plt.show()








