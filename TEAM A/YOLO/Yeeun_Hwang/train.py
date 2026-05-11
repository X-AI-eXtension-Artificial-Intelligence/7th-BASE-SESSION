"""
[파일 개요]
train.py — YOLOv1 학습 파이프라인

[학습 절차 요약]
  1. setup_train()  : 모델, 옵티마이저, 스케줄러, 데이터로더 초기화
  2. init_train()   : 체크포인트/사전학습 가중치 로드 or 처음부터 시작
  3. train()        : MAX_EPOCHS 동안 에폭 루프
       ├─ train_epoch()    : gradient accumulation (SUBDIVISIONS 단위) 학습
       └─ validate_epoch() : 검증셋 손실 계산

[Gradient Accumulation 개념]
  메모리 제약으로 BATCH=64를 한 번에 올리기 어려울 때,
  BATCH // SUBDIVISIONS 크기의 미니배치를 SUBDIVISIONS번 반복하여
  실질적인 배치 크기 64를 시뮬레이션하는 기법.
  → SUBDIVISIONS=8이면 8짝의 mini-batch마다 optimizer.step() 한 번.

[학습률 스케줄]
  Burn-in → 구간별 Scale (MultiStepScaleLR)
  논문의 스케줄을 배치(step) 단위로 재현.
"""

import torch as th
import torchvision.transforms as transforms
import torch.optim as opt
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import VOC_Detection
from transforms import RandomScaleTranslate, Resize, RandomColorJitter, RandomHorizontalFlip, ToYOLOTensor
from model import YOLOv1
from loss import YOLO_Loss
from tqdm import tqdm
from typing import List, Tuple, Dict

# 모델 하이퍼파라미터
S = 7       # 그리드 분할 수 (7×7)
B = 2       # 셀당 예측 박스 수
D = 448     # 입력 이미지 해상도

# 손실 함수 하이퍼파라미터
L_COORD = 5.0   # 위치 손실 가중치 (논문 기본값)
L_NOOBJ = 0.5   # 빈 셀 objectness 손실 가중치 (논문 기본값)

# 데이터 증강 하이퍼파라미터
HUE        = 0.1   # HSV 색상 변화 범위
SATURATION = 1.5   # 채도 변화 배율
EXPOSURE   = 1.5   # 노출(밝기) 변화 배율

RESIZE_PROB   = 0.2   # 단순 리사이즈 확률
ZOOM_OUT_PROB = 0.4   # 축소(줌아웃) 확률
ZOOM_IN_PROB  = 0.4   # 확대(줌인)  확률
JITTER        = 0.2   # 랜덤 크롭/이동 범위 비율

# 데이터 로딩 설정
BATCH       = 64    # 논리적(누적) 배치 크기
SUBDIVISIONS = 8    # 실제 미니배치 분할 수 → 실제 배치 = BATCH // SUBDIVISIONS = 8
NUM_WORKERS  = 10   # 데이터 로딩 병렬 워커 수
SHUFFLE      = True
PIN_MEMORY   = True  # GPU 전송 속도 향상 (CUDA 사용 시)
DROP_LAST    = True  # 마지막 불완전 배치 제거 (BN 안정성)


MAX_EPOCHS = 156
INIT_LR    = 0.0005   
BURN_IN     = 100    
BURN_IN_POW = 2.      

# 각 step(배치 횟수)에서 lr에 scale을 곱하는 스케줄
# (step, scale): step 배치째에 현재 lr * scale
LR_SCHEDULE = [(750,   2.0),
               (1500,  2.0),
               (2250,  1.25),
               (3250,  1.60),
               (5500,  1.25),
               (15000, 0.8),
               (20000, 0.625),
               (25000, 0.8),
               (30000, 0.5),
               (35000, 0.5)]

MOMENTUM     = 0.9
WEIGHT_DECAY = 0.0005

PASCAL_VOC_DIR_PATH = "/media/soul/DATA/cv_datasets/PASCAL_VOC/VOC_Detection"

DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

LOAD_MODEL               = 'pretrain'
PRETRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                           "Detection/checkpoints/pretrained_model_weights.pt"
TRAINING_CHECKPOINT_PATH = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                           "Detection/checkpoints/training_checkpoint.pt"
TRAINED_MODEL_WEIGHTS    = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                           "Detection/checkpoints/trained_model_weights.pt"
CHECKPOINT_T = 10   # N 에폭마다 체크포인트 저장



class MultiStepScaleLR:
    """
    커스텀 학습률 스케줄러

    [학습률 변화 흐름]
    구간 1 (batch < burn_in):
      lr = init_lr * ((batch + 1) / burn_in) ^ pow
      → 처음에는 매우 작은 lr에서 시작해 점진적으로 증가
      → 학습 초반의 gradient 폭발 방지

    구간 2 (batch >= burn_in):
      지정된 step(배치 번호)에 도달하면 lr *= scale
      → 스텝별 배율 조정 (상승/하강 혼재)

    [PyTorch 스케줄러와의 차이]
      th.optim.lr_scheduler 와 달리 에폭 단위가 아닌 배치(step) 단위로 동작.
      또한 상승(warmup)과 하강을 모두 지원하는 사용자 정의 로직.

    [state_dict / load_state_dict]
      optimizer를 제외한 내부 상태를 저장/복원하여 학습 중단 후 재개 가능.
    """
    def __init__(self,
                 optimizer: opt.SGD,
                 init_lr: float,
                 lr_schedule: List[Tuple[int, float]],
                 burn_in: int,
                 burn_in_pow: float) -> None:
        self.optimizer = optimizer
        # zip(*lr_schedule) : [(s1,f1),(s2,f2),...] → (s1,s2,...), (f1,f2,...)
        self.steps, self.scales = zip(*lr_schedule)
        self.burn_in  = burn_in
        self.init_lr  = init_lr
        self.pow      = burn_in_pow
        self.batch    = 0       # 누적 배치 카운터
        self.next_step_ind = 0  # 다음으로 적용할 LR_SCHEDULE 인덱스

    def step(self) -> None:
        self.batch += 1
        if self.batch < self.burn_in:
            # Burn-in: 지수적 증가 ((batch+1)/burn_in)^pow
            self.optimizer.param_groups[0]['lr'] = self.init_lr * ((self.batch + 1) / self.burn_in) ** self.pow
        elif self.next_step_ind < len(self.steps) and self.batch == self.steps[self.next_step_ind]:
            # 스케줄 적용: 현재 lr에 scale 곱셈
            self.optimizer.param_groups[0]['lr'] *= self.scales[self.next_step_ind]
            self.next_step_ind += 1

    def state_dict(self) -> dict:
        return {key: value for (key, value) in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)


def train_epoch(train_loader: DataLoader,
                model: YOLOv1,
                optimizer: opt.SGD,
                criterion: YOLO_Loss,
                scheduler: MultiStepScaleLR,
                mini_batch: int) -> Tuple[float, int]:
    """
    한 에폭 학습 수행.

    [Gradient Accumulation 구현]
      SUBDIVISIONS번의 미니배치마다 optimizer.step() 1회 호출.
      - loss를 SUBDIVISIONS로 나눠서 backward → 누적 gradient가 전체 배치 기준이 됨
      - mini_batch == SUBDIVISIONS가 되면 실제로 파라미터 업데이트 수행

    [에폭 경계 처리]
      mini_batch는 에폭 간에 유지됨 → 에폭이 끊겨도 누적 상태 보존.
      이를 통해 정확한 BATCH 단위 파라미터 업데이트가 가능.

    """
    av_loss = 0.

    model.train()
    for x, y_gt in train_loader:
        mini_batch += 1
        x, y_gt = x.to(DEVICE), y_gt.to(DEVICE)

        y_pred = model(x)
        # SUBDIVISIONS로 나눠 누적 → 실질적 배치 크기에 대한 평균 gradient
        loss = criterion(y_pred, y_gt) / SUBDIVISIONS
        loss.backward()  # gradient 누적 (zero_grad 하지 않음)

        if mini_batch == SUBDIVISIONS:
            # SUBDIVISIONS개 미니배치 누적 후 한 번에 업데이트
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            mini_batch = 0  # 카운터 리셋

        # 로그용 손실은 SUBDIVISIONS를 되돌려 원래 스케일로 기록
        av_loss += loss.item() * SUBDIVISIONS

    av_loss /= len(train_loader)
    return av_loss, mini_batch


def validate_epoch(val_loader: DataLoader,
                   model: YOLOv1,
                   criterion: YOLO_Loss) -> float:
    """
    검증 에폭 수행. gradient 계산 없이 손실만 측정.

    [th.no_grad()]
      autograd 그래프를 생성하지 않아 메모리/연산 절약.
      model.eval()과 함께 사용 → Dropout 비활성화, BN이 running stats 사용.

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
          test_loader: DataLoader,
          model: YOLOv1,
          optimizer: opt.SGD,
          criterion: YOLO_Loss,
          scheduler: MultiStepScaleLR,
          epoch: int,
          mini_batch: int,
          train_loss_history: List[float],
          test_loss_history: List[float]) -> None:

    pbar = tqdm(total=MAX_EPOCHS, desc='Training Epoch', initial=epoch, unit='epoch', position=0, leave=True)

    # mini_batch=0이면 gradient 버퍼 초기화 (재개 시에는 기존 gradient 유지)
    if mini_batch == 0:
        optimizer.zero_grad()

    while epoch < MAX_EPOCHS:
        epoch += 1

        train_loss, mini_batch = train_epoch(train_loader, model, optimizer, criterion, scheduler, mini_batch)
        test_loss = validate_epoch(test_loader, model, criterion)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        # CHECKPOINT_T 에폭마다 체크포인트 저장
        if epoch % CHECKPOINT_T == 0:
            th.save({
                'epoch': epoch,
                'mini_batch': mini_batch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss_history': train_loss_history,
                'test_loss_history': test_loss_history,
                # 현재 누적 gradient 저장: {param_name: grad_tensor}
                'grads': {p[0]: p[1].grad for p in model.named_parameters()}
            }, TRAINING_CHECKPOINT_PATH)

        pbar.set_postfix_str(f'Train Loss={train_loss:.3f}, Test Loss={test_loss:.3f}')
        pbar.update(1)

    # 학습 완료 후 최종 가중치 저장
    th.save(model.state_dict(), TRAINED_MODEL_WEIGHTS)
    pbar.close()


def setup_train():
    """
    학습에 필요한 모든 객체를 초기화하여 반환.

    [옵티마이저 초기 lr 설정]
      burn-in 시작 직전(batch=0)의 lr을 재현하기 위해:
        lr_init = INIT_LR * (1/BURN_IN)^BURN_IN_POW
      MultiStepScaleLR.step()이 첫 호출 시 batch=1로 lr을 갱신하므로
      옵티마이저 생성 시에는 burn-in 공식의 batch=0 값을 사용.

    [데이터 증강]
      학습셋: RandomScaleTranslate → RandomColorJitter → RandomHorizontalFlip → ToYOLOTensor
      검증셋: Resize → ToYOLOTensor (증강 없이 정규화만)

    [WeightedRandomSampler]
      import는 있으나 현재 사용되지 않음.
      클래스 불균형 처리 시 활성화 가능.

    :return: (train_loader, test_loader, model, optimizer, scheduler, criterion)
    """
    model = YOLOv1(S=S, B=B, C=VOC_Detection.C).to(DEVICE)

    optimizer = opt.SGD(
        params=model.parameters(),
        # burn-in 시작 lr: INIT_LR * (1/BURN_IN)^BURN_IN_POW
        lr=INIT_LR * (1 / BURN_IN) ** BURN_IN_POW,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = MultiStepScaleLR(
        optimizer,
        init_lr=INIT_LR,
        lr_schedule=LR_SCHEDULE,
        burn_in=BURN_IN,
        burn_in_pow=BURN_IN_POW
    )

    criterion = YOLO_Loss(S=S, C=VOC_Detection.C, B=B, D=D,
                          L_coord=L_COORD, L_noobj=L_NOOBJ).to(DEVICE)

    # 학습셋: 다양한 데이터 증강 적용 
    train_dataset = VOC_Detection(
        root_dir=PASCAL_VOC_DIR_PATH,
        split='train',
        transforms=transforms.Compose([
            RandomScaleTranslate(output_size=D,
                                 jitter=JITTER,
                                 resize_p=RESIZE_PROB,
                                 zoom_out_p=ZOOM_OUT_PROB,
                                 zoom_in_p=ZOOM_IN_PROB),
            RandomColorJitter(hue=HUE, sat=SATURATION, exp=EXPOSURE),
            RandomHorizontalFlip(p=0.5),
            ToYOLOTensor(S=S, C=VOC_Detection.C,
                         normalize=[[0.4549, 0.4341, 0.4010],   # PASCAL VOC 채널별 평균
                                    [0.2703, 0.2672, 0.2808]])  # PASCAL VOC 채널별 표준편차
        ])
    )

    # 검증셋: 리사이즈 + 정규화만 
    test_dataset = VOC_Detection(
        root_dir=PASCAL_VOC_DIR_PATH,
        split='test',
        transforms=transforms.Compose([
            Resize(output_size=D),
            ToYOLOTensor(S=S, C=VOC_Detection.C,
                         normalize=[[0.4549, 0.4341, 0.4010],
                                    [0.2703, 0.2672, 0.2808]])
        ])
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH // SUBDIVISIONS,  # 실제 미니배치 크기 = 64 // 8 = 8
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=SHUFFLE,
        drop_last=DROP_LAST   # BN 안정성: 마지막 불완전 배치 제거
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH // SUBDIVISIONS,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
        # shuffle=False (기본값): 재현성 보장
    )

    return train_loader, test_loader, model, optimizer, scheduler, criterion


def init_train(model: YOLOv1,
               optimizer: opt.SGD,
               scheduler: MultiStepScaleLR) -> Tuple[int, int, List[float], List[float]]:
    """
    LOAD_MODEL 설정에 따라 세 가지 시작 모드 중 하나를 선택:

    [None]
      완전히 새로 학습. epoch=0, 빈 손실 이력.

    ['pretrain']
      ImageNet 분류 사전학습 가중치만 로드 (strict=False).
      - strict=False: detection head에 없는 레이어(분류 head)는 무시
      - backbone 가중치만 복사되어 탐지 파인튜닝 시작
      - epoch, optimizer, scheduler는 리셋

    ['train']
      이전 체크포인트에서 완전히 재개.
      - epoch, mini_batch, 손실 이력, 모델/옵티마이저/스케줄러 상태 모두 복원
      - gradient도 복원하여 gradient accumulation 이어받기

    :param model:     YOLOv1 탐지 모델
    :param optimizer: SGD 옵티마이저
    :param scheduler: 학습률 스케줄러
    :return: (epoch, mini_batch, train_loss_history, test_loss_history)
    """
    if LOAD_MODEL is None:
        epoch = 0
        mini_batch = 0
        train_loss_history = []
        test_loss_history  = []

    elif LOAD_MODEL == 'pretrain':
        pretrained_model_weights = th.load(PRETRAINED_MODEL_WEIGHTS)
        # strict=False: 분류 head 가중치(분류 모드 전용 레이어)는 무시하고 backbone만 로드
        model.load_state_dict(pretrained_model_weights, strict=False)
        epoch = 0
        mini_batch = 0
        train_loss_history = []
        test_loss_history  = []

    elif LOAD_MODEL == 'train':
        checkpoint = th.load(TRAINING_CHECKPOINT_PATH)

        epoch              = checkpoint['epoch']
        mini_batch         = checkpoint['mini_batch']
        train_loss_history = checkpoint['train_loss_history']
        test_loss_history  = checkpoint['test_loss_history']

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 누적 gradient 복원: {name: grad} → 각 파라미터의 .grad에 할당
        for p in model.named_parameters():
            p[1].grad = checkpoint['grads'][p[0]]

    else:
        assert 0  # 잘못된 LOAD_MODEL 값

    return epoch, mini_batch, train_loss_history, test_loss_history


def main():
    """
    학습 진입점.
    setup → init → train 순서로 실행.
    """
    train_loader, test_loader, model, optimizer, scheduler, criterion = setup_train()
    epoch, mini_batch, train_loss_hist, test_loss_hist = init_train(model, optimizer, scheduler)
    train(train_loader, test_loader, model, optimizer, criterion, scheduler,
          epoch, mini_batch,
          train_loss_hist, test_loss_hist)


if __name__ == '__main__':
    main()