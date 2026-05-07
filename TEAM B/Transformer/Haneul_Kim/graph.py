"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import matplotlib.pyplot as plt
import re

# txt 파일 읽어서 float 리스트로 변환
def read(name):
    f = open(name, 'r')
    file = f.read()
    # 대괄호 제거
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()
    # 문자열 숫자를 float 형태로 변환
    return [float(i) for idx, i in enumerate(file.split(','))]

# loss 또는 bleu 그래프 출력
def draw(mode):
    # loss 그래프
    if mode == 'loss':
        train = read('./result/train_loss.txt')
        test = read('./result/test_loss.txt')
        # train loss
        plt.plot(train, 'r', label='train')
        # validation loss
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

# loss 그래프 출력 + bleu score 그래프 출력
if __name__ == '__main__':
    draw(mode='loss')
    draw(mode='bleu')
