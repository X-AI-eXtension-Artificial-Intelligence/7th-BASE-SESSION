"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import matplotlib.pyplot as plt
import re


def read(name):
    with open(name, 'r') as f:
        file = f.read()
    file = re.sub(r'\[', '', file)
    file = re.sub(r'\]', '', file)
    return [float(i) for i in file.split(',')]


def draw(mode):
    if mode == 'loss':
        train = read('./result/train_loss.txt')
        test  = read('./result/test_loss.txt')
        plt.plot(train, 'r', label='train')
        plt.plot(test,  'b', label='validation')
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
