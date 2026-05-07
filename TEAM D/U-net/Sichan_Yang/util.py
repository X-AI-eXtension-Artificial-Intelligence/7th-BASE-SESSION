import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 저장
def save(ckpt_dir, net, optim, epoch):
    # 체크포인트 저장 디렉토리가 존재하지 않으면 새로 생성
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # 모델과 옵티마이저의 state_dict를 딕셔너리로 저장
    torch.save({'net': net.state_dict(),             # 모델 파라미터 저장
                'optim': optim.state_dict()},        # 옵티마이저 상태 저장
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch)) 

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    # 체크포인트 디렉토리가 존재하지 않을 경우: 에폭 0부터 시작
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    # 디렉토리 내 파일 목록을 불러옴
    ckpt_lst = os.listdir(ckpt_dir)

    # 숫자(에폭 번호)를 기준으로 정렬하여 가장 마지막(최신) 파일 찾기
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # 가장 마지막 체크포인트 파일 로드
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    # 모델 파라미터 및 옵티마이저 상태 복원
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    # 파일명에서 에폭 번호 추출
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    # 모델, 옵티마이저, 에폭 반환
    return net, optim, epoch
