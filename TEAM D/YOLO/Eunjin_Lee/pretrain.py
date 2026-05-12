"""
pretrain.py - YOLOv1 백본 사전학습 (ImageNet 분류)

YOLOv1 논문에서는 객체탐지 학습 전에 백본 네트워크를 ImageNet 1000-class 분류 태스크로
사전학습합니다. 이 파일은 해당 사전학습 과정을 구현합니다.

주요 특징:
  - ImageNet 2012 데이터셋 사용 (1000 클래스)
  - 입력 크기: 224×224 (탐지 시 448×448의 절반)
  - SGD + Momentum + ReduceLROnPlateau 스케줄러
  - Single-crop Top1/Top5 정확도 평가
"""

import torch as th
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as opt
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
import os
from model import YOLOv1
from tqdm import tqdm
from typing import Tuple, List

# ===== 모델 하이퍼파라미터 =====
S = 7       # 그리드 크기 (사전학습에서는 사용되지 않지만 모델 생성에 필요)
B = 2       # 바운딩 박스 수
C = 1000    # ImageNet 클래스 수

# ===== 이미지 변환 하이퍼파라미터 =====
RESIZE_D = 256  # 리사이즈 크기
INPUT_D = 224   # 모델 입력 크기 (224×224)

# ===== 데이터 로딩 하이퍼파라미터 =====
MINI_BATCH = 256    # 미니배치 크기
NUM_WORKERS = 5     # 데이터 로딩 워커 수
SHUFFLE = True      # 학습 데이터 셔플
PIN_MEMORY = True   # GPU 전송 최적화
DROP_LAST = True    # 마지막 불완전 배치 버림

# ===== 학습 하이퍼파라미터 =====
MAX_EPOCHS = 90     # 최대 에폭 수
INIT_LR = 0.1      # 초기 학습률
MOMENTUM = 0.9      # SGD 모멘텀
WEIGHT_DECAY = 0.0001   # L2 정규화 계수
PATIENCE = 2        # 학습률 감소 전 대기 에폭 수
MIN_LR = 0.0001    # 최소 학습률

# ===== 데이터셋 경로 =====
IMAGENET_DIR_PATH = "/home/soul/Development/datasets/ImageNet"

# ===== 디바이스 설정 =====
DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

# ===== 체크포인트 설정 =====
TRAIN_MODEL = False  # False: 저장된 가중치 로드하여 평가만 수행
LOAD_MODEL = True    # True: 체크포인트에서 학습 재개
CHECKPOINT_PATH = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                  "Detection/checkpoints/pretrain_checkpoint.pt"
PRETRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                           "Detection/checkpoints/pretrained_model_weights.pt"
CHECKPOINT_T = 1  # 체크포인트 저장 주기 (에폭)


def train_epoch(train_loader: DataLoader,
                model: YOLOv1,
                optimizer: opt.SGD,
                criterion: nn.CrossEntropyLoss) -> float:
    """
    1 에폭 학습. Cross-Entropy 손실로 분류 모델을 학습합니다.

    :param train_loader: ImageNet 학습 데이터 로더
    :param model: YOLOv1 분류 모델
    :param optimizer: SGD 옵티마이저
    :param criterion: Cross-Entropy 손실 함수
    :return: 에폭 평균 학습 손실
    """
    av_loss = 0.
    model.train()
    for x, y_gt in train_loader:
        x, y_gt = x.to(DEVICE), y_gt.to(DEVICE)
        y_pred = model(x)

        loss = criterion(y_pred, y_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        av_loss += loss.item()

    av_loss /= len(train_loader)
    return av_loss


def validate_epoch(val_loader: DataLoader,
                   model: YOLOv1,
                   criterion: nn.CrossEntropyLoss) -> float:
    """
    1 에폭 검증. 검증 데이터에 대한 평균 손실을 계산합니다.

    :param val_loader: ImageNet 검증 데이터 로더
    :param model: YOLOv1 분류 모델
    :param criterion: Cross-Entropy 손실 함수
    :return: 에폭 평균 검증 손실
    """
    av_loss = 0.
    with th.no_grad():
        model.eval()
        for x, y_gt in val_loader:
            x, y_gt = x.to(DEVICE), y_gt.to(DEVICE)
            y_pred = model(x)
            loss = criterion(y_pred, y_gt)
            av_loss += loss.item()

    av_loss /= len(val_loader)
    return av_loss


def train(train_loader: DataLoader,
          val_loader: DataLoader,
          model: YOLOv1,
          optimizer: opt.SGD,
          scheduler: lr_scheduler.ReduceLROnPlateau,
          criterion: nn.CrossEntropyLoss,
          epoch: int,
          train_loss_history: List,
          val_loss_history: List) -> None:
    """
    전체 사전학습 루프.
    검증 손실이 감소하지 않으면 학습률을 0.1배로 줄입니다 (ReduceLROnPlateau).

    :param train_loader: ImageNet 학습 데이터 로더
    :param val_loader: ImageNet 검증 데이터 로더
    :param model: YOLOv1 분류 모델
    :param optimizer: SGD 옵티마이저
    :param scheduler: ReduceLROnPlateau 스케줄러
    :param criterion: Cross-Entropy 손실 함수
    :param epoch: 시작 에폭
    :param train_loss_history: 학습 손실 기록 리스트
    :param val_loss_history: 검증 손실 기록 리스트
    """
    pbar = tqdm(total=MAX_EPOCHS, desc='Training Epoch', initial=epoch, unit='epoch')
    while epoch < MAX_EPOCHS:
        epoch += 1

        train_loss = train_epoch(train_loader, model, optimizer, criterion)
        val_loss = validate_epoch(val_loader, model, criterion)
        scheduler.step(val_loss)  # 검증 손실 기반 학습률 조정

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # 주기적 체크포인트 저장
        if epoch % CHECKPOINT_T == 0:
            th.save({'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(),
                     'train_loss_history': train_loss_history,
                     'val_loss_history': val_loss_history
                     }, CHECKPOINT_PATH)

        pbar.set_postfix_str(f'Train Loss={train_loss:.3f}, Val Loss={val_loss:.3f}')
        pbar.update(1)
    pbar.close()


def measure_accuracy(model: YOLOv1, val_loader: DataLoader) -> Tuple[float, float]:
    """
    ImageNet 검증 세트에서 Single-crop Top1/Top5 정확도를 측정합니다.

    - Top1: 최고 확률 예측이 정답인 비율
    - Top5: 상위 5개 예측 중 정답이 포함된 비율

    :param model: YOLOv1 분류 모델
    :param val_loader: ImageNet 검증 데이터 로더
    :return: (Top1 정확도, Top5 정확도)
    """
    top1_accuracy = 0.
    top5_accuracy = 0.
    total = 0
    with th.no_grad():
        model.eval()
        for x, y_gt in val_loader:
            x, y_gt = x.to(DEVICE), y_gt.to(DEVICE)
            y_pred = model(x)

            _, top5_preds = th.topk(y_pred, k=5, dim=-1)
            top1_accuracy += th.sum(top5_preds[:, 0] == y_gt).item()
            top5_accuracy += th.sum(top5_preds == y_gt.reshape(-1, 1)).item()
            total += x.shape[0]

        top1_accuracy /= total
        top5_accuracy /= total

        return top1_accuracy, top5_accuracy


def setup_train() -> Tuple[DataLoader,
                           DataLoader,
                           YOLOv1,
                           opt.SGD,
                           lr_scheduler.ReduceLROnPlateau,
                           nn.CrossEntropyLoss]:
    """
    사전학습에 필요한 구성요소를 초기화합니다.

    데이터 증강:
      학습: Resize(256) → CenterCrop(256) → RandomCrop(224) → RandomHFlip → ColorJitter → Normalize → RandomErasing
      검증: Resize(256) → CenterCrop(224) → Normalize

    :return: (train_loader, val_loader, model, optimizer, scheduler, criterion)
    """
    # 분류 모드로 모델 생성 (백본 + AvgPool + FC)
    model = YOLOv1(S=S,
                   B=B,
                   C=C,
                   mode='classification').to(DEVICE)

    optimizer = opt.SGD(params=model.parameters(),
                        lr=INIT_LR,
                        momentum=MOMENTUM,
                        weight_decay=WEIGHT_DECAY)

    # 검증 손실이 PATIENCE 에폭 동안 감소하지 않으면 학습률 × 0.1
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=PATIENCE, min_lr=MIN_LR)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # ImageNet 학습 데이터셋 (다양한 증강 적용)
    train_dataset = ImageNet(root=IMAGENET_DIR_PATH,
                             split='train',
                             transform=transforms.Compose([transforms.Resize(RESIZE_D),
                                                           transforms.CenterCrop(RESIZE_D),
                                                           transforms.RandomCrop(INPUT_D),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.ColorJitter(),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225]),
                                                           transforms.RandomErasing()]))

    # ImageNet 검증 데이터셋 (증강 없이 중앙 크롭만)
    val_dataset = ImageNet(root=IMAGENET_DIR_PATH,
                           split='val',
                           transform=transforms.Compose([transforms.Resize(RESIZE_D),
                                                         transforms.CenterCrop(INPUT_D),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])]))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=MINI_BATCH,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              shuffle=SHUFFLE,
                              drop_last=DROP_LAST)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=MINI_BATCH,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY)

    return train_loader, val_loader, model, optimizer, scheduler, criterion


def init_train(model: YOLOv1,
               optimizer: opt.SGD,
               scheduler: lr_scheduler.ReduceLROnPlateau) -> Tuple[int, List[float], List[float]]:
    """
    학습 상태 초기화.
      - TRAIN_MODEL=True, LOAD_MODEL=True: 체크포인트에서 학습 재개
      - TRAIN_MODEL=True, LOAD_MODEL=False: 처음부터 학습
      - TRAIN_MODEL=False: 저장된 가중치 로드 (평가만 수행)

    :param model: YOLOv1 분류 모델
    :param optimizer: SGD 옵티마이저
    :param scheduler: 학습률 스케줄러
    :return: (현재 에폭, 학습 손실 기록, 검증 손실 기록)
    """
    if TRAIN_MODEL:
        if LOAD_MODEL and os.path.exists(CHECKPOINT_PATH):
            # 체크포인트에서 모든 상태 복원
            checkpoint = th.load(CHECKPOINT_PATH)

            epoch = checkpoint['epoch']
            train_loss_history = checkpoint['train_loss_history']
            val_loss_history = checkpoint['val_loss_history']

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # 처음부터 학습
            epoch = 0
            train_loss_history = []
            val_loss_history = []
    else:
        # 평가 모드: 사전학습 완료된 가중치 로드
        pretrained_model_weights = th.load(PRETRAINED_MODEL_WEIGHTS)
        model.load_state_dict(pretrained_model_weights)

        epoch, train_loss_history, val_loss_history = None, None, None

    return epoch, train_loss_history, val_loss_history


def main():
    """메인 함수: 설정 → 초기화 → (학습) → 정확도 평가"""
    train_loader, val_loader, model, optimizer, scheduler, criterion = setup_train()
    epoch, train_loss_history, val_loss_history = init_train(model, optimizer, scheduler)
    if TRAIN_MODEL:
        train(train_loader, val_loader, model, optimizer, scheduler, criterion, epoch,
              train_loss_history, val_loss_history)
    top1_accuracy, top5_accuracy = measure_accuracy(model, val_loader)

    print(f'Single-Crop Top1 Accuracy = {top1_accuracy * 100:.2f}%')
    print(f'Single-Crop Top5 Accuracy = {top5_accuracy * 100:.2f}%')

    if TRAIN_MODEL:
        print(f'Train Loss History: {train_loss_history}')
        print(f'Validation Loss History: {val_loss_history}')


if __name__ == '__main__':
    main()
