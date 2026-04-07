import os
import numpy as np
import matplotlib.pyplot as plt

## 결과 파일이 저장된 경로 설정
result_dir = r'C:\Users\이윤정\OneDrive - KookminUNIV\바탕 화면\2026-1 학교생활\X_Ai'

# 해당 디렉토리에 있는 모든 파일 목록을 가져옴
lst_data = os.listdir(result_dir)

# 파일명 시작 부분에 따라 label, input, output 리스트로 분류
lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]

# 파일들을 이름순으로 정렬
lst_label.sort()
lst_input.sort()
lst_output.sort()

## 시각화할 데이터 선택
id = 0

label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

## 입력 이미지/ 정답 이미지/ 모델 출력 이미지(예측 결과)
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








