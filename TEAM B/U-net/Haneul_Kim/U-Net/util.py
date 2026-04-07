## 모델과 optimizer 상태 저장하고 불러오는 기능 제공

import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 저장하기
# 모델과 optimizer 상태를 checkpoint 파일로 저장하는 함수
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
# 저장된 checkpoint를 불러와 모델과 optimizer 상태를 복원하는 함수
# checkpoint를 활용하여 학습을 중단한 지점부터 이어서 학습 가능
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch
