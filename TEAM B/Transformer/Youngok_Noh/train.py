"""
graph.py
- 학습 loss와 validation loss를 그래프로 저장하는 보조 파일입니다.
"""

import matplotlib.pyplot as plt


def draw(mode, train, val, path):
    """
    mode: 그래프 제목에 들어갈 이름입니다. 예: 'loss'
    train: epoch별 train loss 리스트
    val: epoch별 validation loss 리스트
    path: 저장할 이미지 파일 경로
    """
    plt.plot(train, 'r', label='train')
    plt.plot(val, 'b', label='validation')
    plt.title(mode)
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.savefig(path)
    plt.close()
