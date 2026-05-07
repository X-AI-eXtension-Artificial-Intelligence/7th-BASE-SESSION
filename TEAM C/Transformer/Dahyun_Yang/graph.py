"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import matplotlib.pyplot as plt
import re


def read(name):
    """
    txt 파일에 저장된 숫자 리스트를 읽어와서
    Python의 float 리스트로 변환하는 함수

    예:
    파일 내용이 "[1.2, 0.9, 0.7]" 이라면
    반환값은 [1.2, 0.9, 0.7]
    """

     # name 경로에 있는 파일을 읽기 모드로 연다
    f = open(name, 'r')
    file = f.read()

     # 문자열 안의 '[, ]' 문자를 제거한다
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]


def draw(mode):
    """
    mode 값에 따라 loss 그래프 또는 bleu score 그래프를 그리는 함수

    mode='loss'이면 train loss와 validation loss를 그림
    mode='bleu'이면 bleu score를 그림
    """
    if mode == 'loss':
        train = read('./result/train_loss.txt')
        test = read('./result/test_loss.txt')
        plt.plot(train, 'r', label='train')
        plt.plot(test, 'b', label='validation')
        plt.legend(loc='lower left')


    elif mode == 'bleu':
        bleu = read('./result/bleu.txt')
        plt.plot(bleu, 'b', label='bleu score')
        plt.legend(loc='lower right')

    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.title('training result')
    plt.grid(True, which='both', axis='both')
    plt.show()


if __name__ == '__main__':
    draw(mode='loss')
    draw(mode='bleu')
