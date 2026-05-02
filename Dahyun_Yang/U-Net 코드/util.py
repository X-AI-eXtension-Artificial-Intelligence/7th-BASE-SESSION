import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 저장하기(함수의 현재 상태 저장하지, 학습 중인 모델의 가중치)
def save(ckpt_dir, net, optim, epoch):
    # 저장할 폴더가 없으면 새로 만듭니다
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    # 파이토치 전용 저장 방식: 딕셔너리 형태로 묶어서 저장
    # net.state_dict(): 모델의 가중치(Weight)들
    # optim.state_dict(): 학습률 등 Optimizer의 현재 상태 정보
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기(저장된 파일 중 가장 마지막 최신의 파일을 찾아 모델에 이식함)
def load(ckpt_dir, net, optim):
    # 저장된 폴더가 아예 없으면 처음부터 시작 (epoch=0)
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
    # 저장된 파일 목록을 가져와서 숫자 순서대로 정렬
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    # 가장 마지막(최신) 체크포인트 파일을 불러옴
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    # 불러온 값을 현재 모델과 Optimizer에 덮어씌움
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    # 파일 이름에서 숫자만 추출해 현재 몇 에폭까지 공부했는지 확인
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

# 중요한 이유
# 1. 불의의 사고에 대비하여 학습하는 과정에 날아가도 마지막으로 저장된 시점 불러올 수 있음
# 2. 최적의 시점 선택 가능 → 100번 학습 시켰는데 70번째가 가장 성능이 좋았다면 해당 시점의 파일 불러오기 가능
# 3. 학습과 데이터의 분리 → 각각 학습 및 훈련을 시켜서 한번 학습을 돌렸으면 굳이 학습 없이 바로 예측 결과 뽑아낼 수 있음