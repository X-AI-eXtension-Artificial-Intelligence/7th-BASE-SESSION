import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 가을님이 알려주신 절대 경로로 직접 설정
result_dir = '/Users/lotte/Documents/GitHub/youtube-cnn-002-pytorch-unet/autumn/results/numpy'

# 폴더 존재 여부 확인 (디버깅용)
if not os.path.exists(result_dir):
    print(f"❌ 에러: 폴더를 찾을 수 없습니다 -> {result_dir}")
    # 혹시 numpy 폴더가 results 바로 아래에 없는지 확인하기 위해 목록 출력
    parent_dir = '/Users/lotte/Documents/GitHub/youtube-cnn-002-pytorch-unet/autumn/results'
    if os.path.exists(parent_dir):
        print(f"📂 현재 results 폴더 안의 목록: {os.listdir(parent_dir)}")
else:
    # 2. 파일 목록 읽기 및 숫자 순서 정렬
    lst_data = os.listdir(result_dir)
    
    # 파일명에서 숫자만 추출해 정렬 (label_0000.npy 등을 순서대로 정렬)
    lst_label = sorted([f for f in lst_data if f.startswith('label')], key=lambda f: int(''.join(filter(str.isdigit, f))))
    lst_input = sorted([f for f in lst_data if f.startswith('input')], key=lambda f: int(''.join(filter(str.isdigit, f))))
    lst_output = sorted([f for f in lst_data if f.startswith('output')], key=lambda f: int(''.join(filter(str.isdigit, f))))

    # 3. 안전하게 화면에 출력 (최대 4개 세트)
    num_display = min(len(lst_label), 4)

    if num_display == 0:
        print("⚠️ 시각화할 데이터가 없습니다. 폴더 내 파일명을 확인해 주세요.")
    else:
        plt.figure(figsize=(12, 3 * num_display))
        for i in range(num_display):
            # 인덱스 에러 방지를 위해 실제 리스트 인덱스 i 사용
            label = np.load(os.path.join(result_dir, lst_label[i]))
            input = np.load(os.path.join(result_dir, lst_input[i]))
            output = np.load(os.path.join(result_dir, lst_output[i]))

            # 시각화 (Input, Label, Output 순서)
            plt.subplot(num_display, 3, 3*i + 1)
            plt.imshow(input.squeeze(), cmap='gray'); plt.title(f'Input {i}'); plt.axis('off')

            plt.subplot(num_display, 3, 3*i + 2)
            plt.imshow(label.squeeze(), cmap='gray'); plt.title(f'Label {i}'); plt.axis('off')

            plt.subplot(num_display, 3, 3*i + 3)
            plt.imshow(output.squeeze(), cmap='gray'); plt.title(f'Output {i}'); plt.axis('off')

        plt.tight_layout()
        plt.show()