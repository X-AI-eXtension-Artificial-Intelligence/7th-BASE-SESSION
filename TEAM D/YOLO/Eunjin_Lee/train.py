"""
train.py - YOLOv1 객체탐지 모델 학습 (Fine-tuning)

ImageNet으로 사전학습된 백본을 PASCAL VOC 데이터셋에서 객체탐지 태스크로 미세조정합니다.

주요 특징:
  - Gradient Accumulation: SUBDIVISIONS로 나눠 큰 배치 효과 달성
  - 커스텀 학습률 스케줄러 (MultiStepScaleLR): burn-in + 단계별 스케일링
  - 데이터 증강: RandomScaleTranslate, RandomColorJitter, RandomHorizontalFlip
  - 체크포인트 저장/복원으로 학습 재개 지원
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

# ===== 모델 하이퍼파라미터 =====
S = 7       # 그리드 크기 (7×7)
B = 2       # 셀당 바운딩 박스 수
D = 448     # 입력 이미지 크기 (448×448)

# ===== 손실 함수 하이퍼파라미터 =====
L_COORD = 5.0   # 위치 손실 가중치 (바운딩 박스 정확도 강조)
L_NOOBJ = 0.5   # 비객체 셀 신뢰도 손실 가중치 (학습 안정성)

# ===== 데이터 증강 하이퍼파라미터 =====
HUE = 0.1           # 색조 변화 범위
SATURATION = 1.5    # 채도 변화 범위
EXPOSURE = 1.5      # 노출 변화 범위

RESIZE_PROB = 0.2   # 단순 리사이즈 확률
ZOOM_OUT_PROB = 0.4 # 축소 확률
ZOOM_IN_PROB = 0.4  # 확대 확률
JITTER = 0.2        # 스케일/이동 지터 계수

# ===== 데이터 로딩 하이퍼파라미터 =====
BATCH = 64              # 전체 배치 크기
SUBDIVISIONS = 8        # Gradient Accumulation 분할 수 (실제 미니배치 = BATCH/SUBDIVISIONS = 8)
NUM_WORKERS = 10        # 데이터 로딩 워커 수
SHUFFLE = True          # 학습 데이터 셔플
PIN_MEMORY = True       # GPU 전송 최적화
DROP_LAST = True        # 마지막 불완전 배치 버림

# ===== 학습 하이퍼파라미터 =====
MAX_EPOCHS = 156        # 최대 에폭 수
INIT_LR = 0.0005       # burn-in 후 초기 학습률
BURN_IN = 100          # burn-in 배치 수 (학습률 점진적 증가)
BURN_IN_POW = 2.       # burn-in 학습률 증가 지수
# 학습률 스케줄: (스텝, 스케일 배수)
LR_SCHEDULE = [(750, 2.0),
               (1500, 2.0),
               (2250, 1.25),
               (3250, 1.60),
               (5500, 1.25),
               (15000, 0.8),
               (20000, 0.625),
               (25000, 0.8),
               (30000, 0.5),
               (35000, 0.5)]
MOMENTUM = 0.9          # SGD 모멘텀
WEIGHT_DECAY = 0.0005   # L2 정규화 계수

# ===== 데이터셋 경로 =====
PASCAL_VOC_DIR_PATH = "/media/soul/DATA/cv_datasets/PASCAL_VOC/VOC_Detection"

# ===== 디바이스 설정 =====
DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

# ===== 체크포인트 설정 =====
LOAD_MODEL = 'pretrain'  # 'pretrain': 사전학습 가중치 로드, 'train': 학습 체크포인트 로드, None: 처음부터
PRETRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                           "Detection/checkpoints/pretrained_model_weights.pt"
TRAINING_CHECKPOINT_PATH = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                           "Detection/checkpoints/training_checkpoint.pt"
TRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                        "Detection/checkpoints/trained_model_weights.pt"
CHECKPOINT_T = 10  # 체크포인트 저장 주기 (에폭)


##########################################################################

class MultiStepScaleLR:
    """
    커스텀 학습률 스케줄러.

    동작 방식:
      1. Burn-in 단계: 학습률을 0에서 init_lr까지 점진적으로 증가
         lr = init_lr × (batch/burn_in)^pow
      2. 이후: 지정된 스텝에서 학습률에 스케일 배수를 곱함
    """
    def __init__(self,
                 optimizer: opt.SGD,
                 init_lr: float,
                 lr_schedule: List[Tuple[int, float]],
                 burn_in: int,
                 burn_in_pow: float) -> None:
        """
        :param optimizer: SGD 옵티마이저
        :param init_lr: burn-in 완료 후 학습률
        :param lr_schedule: [(스텝, 스케일 배수), ...] 리스트
        :param burn_in: burn-in 배치 수
        :param burn_in_pow: burn-in 학습률 증가 지수
        """
        self.optimizer = optimizer
        self.steps, self.scales = zip(*lr_schedule)
        self.burn_in = burn_in
        self.init_lr = init_lr
        self.pow = burn_in_pow
        self.batch = 0              # 가중치 업데이트 횟수 카운터
        self.next_step_ind = 0      # 다음 스케줄 스텝 인덱스

    def step(self) -> None:
        """
        가중치 업데이트 시마다 호출하여 학습률을 조정합니다.
        """
        self.batch += 1
        if self.batch < self.burn_in:
            # Burn-in: 학습률 점진적 증가
            self.optimizer.param_groups[0]['lr'] = self.init_lr * ((self.batch+1)/self.burn_in)**self.pow
        elif self.next_step_ind < len(self.steps) and self.batch == self.steps[self.next_step_ind]:
            # 지정된 스텝에서 학습률 스케일링
            self.optimizer.param_groups[0]['lr'] *= self.scales[self.next_step_ind]
            self.next_step_ind += 1

    def state_dict(self) -> dict:
        """
        체크포인트 저장용 상태 딕셔너리 반환 (optimizer 제외).
        """
        return {key: value for (key, value) in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: dict) -> None:
        """
        체크포인트에서 상태를 복원합니다.
        """
        self.__dict__.update(state_dict)


def train_epoch(train_loader: DataLoader,
                model: YOLOv1,
                optimizer: opt.SGD,
                criterion: YOLO_Loss,
                scheduler: MultiStepScaleLR,
                mini_batch: int) -> Tuple[float, int]:
    """
    1 에폭 학습. Gradient Accumulation을 사용하여 큰 배치 효과를 달성합니다.
    SUBDIVISIONS번의 forward/backward 후 한 번 optimizer.step() 수행.

    :param train_loader: PASCAL VOC 학습 데이터 로더
    :param model: YOLOv1 탐지 모델
    :param optimizer: SGD 옵티마이저
    :param criterion: YOLO 손실 함수
    :param scheduler: 학습률 스케줄러
    :param mini_batch: 현재 미니배치 카운터
    :return: (에폭 평균 손실, 업데이트된 미니배치 카운터)
    """
    av_loss = 0.

    model.train()
    for x, y_gt in train_loader:
        mini_batch += 1
        x, y_gt = x.to(DEVICE), y_gt.to(DEVICE)
        y_pred = model(x)
        # 손실을 SUBDIVISIONS로 나눠 gradient 누적
        loss = criterion(y_pred, y_gt) / SUBDIVISIONS
        loss.backward()

        # SUBDIVISIONS번 누적 후 가중치 업데이트
        if mini_batch == SUBDIVISIONS:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            mini_batch = 0

        av_loss += loss.item() * SUBDIVISIONS

    av_loss /= len(train_loader)
    return av_loss, mini_batch


def validate_epoch(val_loader: DataLoader,
                   model: YOLOv1,
                   criterion: YOLO_Loss) -> float:
    """
    1 에폭 검증. 테스트 데이터에 대한 평균 손실을 계산합니다.

    :param val_loader: PASCAL VOC 테스트 데이터 로더
    :param model: YOLOv1 탐지 모델
    :param criterion: YOLO 손실 함수
    :return: 에폭 평균 테스트 손실
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
    """
    전체 학습 루프. MAX_EPOCHS까지 학습하며 주기적으로 체크포인트를 저장합니다.
    학습 완료 후 최종 모델 가중치를 저장합니다.

    :param train_loader: 학습 데이터 로더
    :param test_loader: 테스트 데이터 로더
    :param model: YOLOv1 모델
    :param optimizer: SGD 옵티마이저
    :param criterion: YOLO 손실 함수
    :param scheduler: 학습률 스케줄러
    :param epoch: 시작 에폭
    :param mini_batch: 미니배치 카운터
    :param train_loss_history: 학습 손실 기록
    :param test_loss_history: 테스트 손실 기록
    """
    pbar = tqdm(total=MAX_EPOCHS, desc='Training Epoch', initial=epoch, unit='epoch', position=0, leave=True)
    if mini_batch == 0:
        optimizer.zero_grad()

    while epoch < MAX_EPOCHS:
        epoch += 1

        train_loss, mini_batch = train_epoch(train_loader, model, optimizer, criterion, scheduler, mini_batch)
        test_loss = validate_epoch(test_loader, model, criterion)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        # 주기적 체크포인트 저장 (gradient 포함하여 정확한 재개 가능)
        if epoch % CHECKPOINT_T == 0:
            th.save({'epoch': epoch,
                     'mini_batch': mini_batch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(),
                     'train_loss_history': train_loss_history,
                     'test_loss_history': test_loss_history,
                     'grads': {p[0]: p[1].grad for p in model.named_parameters()}
                     }, TRAINING_CHECKPOINT_PATH)

        pbar.set_postfix_str(f'Train Loss={train_loss:.3f}, Test Loss={test_loss:.3f}')
        pbar.update(1)

    # 학습 완료 후 최종 모델 가중치 저장
    th.save(model.state_dict(), TRAINED_MODEL_WEIGHTS)
    pbar.close()


def setup_train():
    """
    학습에 필요한 모든 구성요소를 초기화합니다:
    모델, 옵티마이저, 스케줄러, 손실함수, 데이터셋, 데이터로더.

    데이터 증강 파이프라인:
      학습: RandomScaleTranslate → RandomColorJitter → RandomHorizontalFlip → ToYOLOTensor
      테스트: Resize → ToYOLOTensor (증강 없음)

    :return: (train_loader, test_loader, model, optimizer, scheduler, criterion)
    """
    model = YOLOv1(S=S,
                   B=B,
                   C=VOC_Detection.C).to(DEVICE)

    # SGD with Momentum (초기 학습률은 burn-in 시작값으로 설정)
    optimizer = opt.SGD(params=model.parameters(),
                        lr=INIT_LR * (1/BURN_IN)**BURN_IN_POW,
                        momentum=MOMENTUM,
                        weight_decay=WEIGHT_DECAY)

    scheduler = MultiStepScaleLR(optimizer,
                                 init_lr=INIT_LR,
                                 lr_schedule=LR_SCHEDULE,
                                 burn_in=BURN_IN,
                                 burn_in_pow=BURN_IN_POW)

    criterion = YOLO_Loss(S=S,
                          C=VOC_Detection.C,
                          B=B,
                          D=D,
                          L_coord=L_COORD,
                          L_noobj=L_NOOBJ).to(DEVICE)

    # 학습 데이터셋: 데이터 증강 적용
    train_dataset = VOC_Detection(root_dir=PASCAL_VOC_DIR_PATH,
                                  split='train',
                                  transforms=transforms.Compose([
                                      RandomScaleTranslate(output_size=D,
                                                           jitter=JITTER,
                                                           resize_p=RESIZE_PROB,
                                                           zoom_out_p=ZOOM_OUT_PROB,
                                                           zoom_in_p=ZOOM_IN_PROB),
                                      RandomColorJitter(hue=HUE,
                                                        sat=SATURATION,
                                                        exp=EXPOSURE),
                                      RandomHorizontalFlip(p=0.5),
                                      ToYOLOTensor(S=S,
                                                   C=VOC_Detection.C,
                                                   normalize=[[0.4549, 0.4341, 0.4010],
                                                              [0.2703, 0.2672, 0.2808]])]))

    # 테스트 데이터셋: 증강 없이 리사이즈만 적용
    test_dataset = VOC_Detection(root_dir=PASCAL_VOC_DIR_PATH,
                                 split='test',
                                 transforms=transforms.Compose([
                                     Resize(output_size=D),
                                     ToYOLOTensor(S=S,
                                                  C=VOC_Detection.C,
                                                  normalize=[[0.4549, 0.4341, 0.4010],
                                                             [0.2703, 0.2672, 0.2808]])]))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH // SUBDIVISIONS,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              shuffle=SHUFFLE,
                              drop_last=DROP_LAST)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH // SUBDIVISIONS,
                             num_workers=NUM_WORKERS,
                             pin_memory=PIN_MEMORY)

    return train_loader, test_loader, model, optimizer, scheduler, criterion


def init_train(model: YOLOv1,
               optimizer: opt.SGD,
               scheduler: MultiStepScaleLR) -> Tuple[int, int, List[float], List[float]]:
    """
    학습 상태 초기화. LOAD_MODEL 설정에 따라:
      - None: 처음부터 학습
      - 'pretrain': 사전학습 가중치 로드 후 처음부터 미세조정
      - 'train': 이전 학습 체크포인트에서 재개

    :param model: YOLOv1 모델
    :param optimizer: SGD 옵티마이저
    :param scheduler: 학습률 스케줄러
    :return: (에폭, 미니배치 카운터, 학습 손실 기록, 테스트 손실 기록)
    """
    if LOAD_MODEL is None:
        epoch = 0
        mini_batch = 0
        train_loss_history = []
        test_loss_history = []

    elif LOAD_MODEL == 'pretrain':
        # 사전학습 가중치 로드 (탐지 헤드는 strict=False로 무시)
        pretrained_model_weights = th.load(PRETRAINED_MODEL_WEIGHTS)
        model.load_state_dict(pretrained_model_weights, strict=False)
        epoch = 0
        mini_batch = 0
        train_loss_history = []
        test_loss_history = []

    elif LOAD_MODEL == 'train':
        # 학습 체크포인트에서 모든 상태 복원
        checkpoint = th.load(TRAINING_CHECKPOINT_PATH)

        epoch = checkpoint['epoch']
        mini_batch = checkpoint['mini_batch']
        train_loss_history = checkpoint['train_loss_history']
        test_loss_history = checkpoint['test_loss_history']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # gradient도 복원하여 정확한 재개
        for p in model.named_parameters():
            p[1].grad = checkpoint['grads'][p[0]]

    else:
        assert 0

    return epoch, mini_batch, train_loss_history, test_loss_history


def main():
    """메인 함수: 학습 설정 → 초기화 → 학습 실행"""
    train_loader, test_loader, model, optimizer, scheduler, criterion = setup_train()
    epoch, mini_batch, train_loss_hist, test_loss_hist = init_train(model, optimizer, scheduler)
    train(train_loader, test_loader, model, optimizer, criterion, scheduler,
          epoch, mini_batch,
          train_loss_hist, test_loss_hist)


if __name__ == '__main__':
    main()
