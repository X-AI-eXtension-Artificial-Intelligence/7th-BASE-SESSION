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

# 하이퍼파라미터 설정

# ── 모델 파라미터 ──────────────────────────────────────────────
S = 7          # 그리드 크기: 이미지를 7×7 = 49개 셀로 분할
B = 2          # 셀당 예측 바운딩 박스 수
D = 448        # 입력 이미지 크기 (448×448 픽셀)

# ── 손실 함수 가중치 ───────────────────────────────────────────
L_COORD = 5.0  # 위치(localization) 손실 가중치 — 위치 오차를 5배 강조
L_NOOBJ = 0.5  # 객체 없는 셀의 confidence 손실 가중치 — 배경 억제

# ── 데이터 증강 파라미터 ────────────────────────────────────────
HUE        = 0.1    # 색조(Hue) 변화 범위 [-0.1, 0.1]
SATURATION = 1.5    # 채도 배율 범위 [1/1.5, 1.5]
EXPOSURE   = 1.5    # 노출 배율 범위 [1/1.5, 1.5]

RESIZE_PROB   = 0.2  # 단순 리사이즈 확률
ZOOM_OUT_PROB = 0.4  # Zoom out 확률
ZOOM_IN_PROB  = 0.4  # Zoom in 확률
JITTER        = 0.2  # 스케일/이동 강도

# ── 데이터 로더 파라미터 ────────────────────────────────────────
BATCH       = 64    # 논리적 배치 크기 (그래디언트 누적 기준)
SUBDIVISIONS = 8    # 실제 미니배치 크기 = BATCH / SUBDIVISIONS = 8
                    # 메모리 절약: 8장씩 forward하고 8번 누적 후 1번 update
NUM_WORKERS = 10    # DataLoader 병렬 처리 워커 수
SHUFFLE     = True
PIN_MEMORY  = True
DROP_LAST   = True  # 마지막 불완전 배치 버림

# ── 학습 하이퍼파라미터 ─────────────────────────────────────────
MAX_EPOCHS = 156    # 총 학습 에폭 수

INIT_LR      = 0.0005   # Burn-in 후 도달하는 기준 학습률
BURN_IN      = 100      # 처음 100 배치는 학습률을 점진적으로 증가
BURN_IN_POW  = 2.       # 학습률 증가 곡선의 지수 (2차 함수적 증가)

# 학습률 스케줄: (step 번호, 학습률 배율) 목록
# step은 optimizer.step() 호출 횟수 기준
LR_SCHEDULE = [
    (750,   2.0),   # 750번째 스텝: 학습률 × 2.0
    (1500,  2.0),
    (2250,  1.25),
    (3250,  1.60),
    (5500,  1.25),
    (15000, 0.8),
    (20000, 0.625),
    (25000, 0.8),
    (30000, 0.5),
    (35000, 0.5),
]

MOMENTUM     = 0.9      # SGD 모멘텀
WEIGHT_DECAY = 0.0005   # L2 정규화 계수

# ── 경로 설정 ── 학습 전 본인 환경에 맞게 수정 필요 ─────────────
PASCAL_VOC_DIR_PATH = "/media/soul/DATA/cv_datasets/PASCAL_VOC/VOC_Detection"

DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

# ── 체크포인트 설정 ────────────────────────────────────────────
# LOAD_MODEL 값에 따라 학습 초기화 방식 결정:
#   'pretrain' : ImageNet 사전학습 가중치를 backbone에 로드 후 detection 학습 시작
#   'train'    : 이전 학습 체크포인트에서 재개
#   None       : 처음부터 학습 (랜덤 초기화)
LOAD_MODEL = 'pretrain'

PRETRAINED_MODEL_WEIGHTS  = "/path/to/checkpoints/pretrained_model_weights.pt"
TRAINING_CHECKPOINT_PATH  = "/path/to/checkpoints/training_checkpoint.pt"
TRAINED_MODEL_WEIGHTS     = "/path/to/checkpoints/trained_model_weights.pt"
CHECKPOINT_T = 10  # 몇 에폭마다 체크포인트를 저장할지


# MultiStepScaleLR: 커스텀 학습률 스케줄러
#   - Burn-in 구간: 처음 burn_in 스텝 동안 학습률을 점진적으로 증가
#   - 이후: 지정된 스텝에서 학습률을 지정된 배율로 조정

class MultiStepScaleLR:

    def __init__(self, optimizer, init_lr, lr_schedule, burn_in, burn_in_pow):
        """
        optimizer    : SGD 옵티마이저
        init_lr      : Burn-in 완료 후 목표 학습률
        lr_schedule  : [(step, scale), ...] 형태의 학습률 조정 스케줄
        burn_in      : 학습률을 점진적으로 올릴 초기 스텝 수
        burn_in_pow  : 학습률 증가 곡선 지수 (클수록 더 완만하게 시작)
        """
        self.optimizer    = optimizer
        self.steps, self.scales = zip(*lr_schedule)  # 스텝과 배율 분리
        self.burn_in   = burn_in
        self.init_lr   = init_lr
        self.pow       = burn_in_pow
        self.batch     = 0   # 현재까지의 optimizer.step() 호출 횟수
        self.next_step_ind = 0  # 다음에 적용할 LR_SCHEDULE 인덱스

    def step(self):
        """
        optimizer.step()이 호출될 때마다 학습률을 업데이트:
          - burn_in 이전: lr = init_lr × ((batch+1)/burn_in)^pow  (점진적 증가)
          - 이후 지정 스텝: lr = 현재 lr × scale
        """
        self.batch += 1
        if self.batch < self.burn_in:
            # Burn-in: 2차 함수적으로 학습률 증가
            self.optimizer.param_groups[0]['lr'] = self.init_lr * ((self.batch + 1) / self.burn_in) ** self.pow
        elif self.next_step_ind < len(self.steps) and self.batch == self.steps[self.next_step_ind]:
            # 지정 스텝에서 학습률 배율 적용
            self.optimizer.param_groups[0]['lr'] *= self.scales[self.next_step_ind]
            self.next_step_ind += 1

    def state_dict(self):
        # 체크포인트 저장용: optimizer를 제외한 멤버 변수 딕셔너리 반환
        return {key: value for (key, value) in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        # 체크포인트 로드: 저장된 상태를 복원
        self.__dict__.update(state_dict)


# ============================================================
# train_epoch: 한 에폭 동안 학습 수행
# ============================================================
def train_epoch(train_loader, model, optimizer, criterion, scheduler, mini_batch):
    """
    그래디언트 누적(Gradient Accumulation):
      SUBDIVISIONS번 미니배치 forward/backward 후 한 번 optimizer.step()
      → 실질적으로 BATCH 크기의 배치로 학습 (메모리 절약)

    반환: (에폭 평균 손실, 업데이트된 mini_batch 카운터)
    """
    av_loss = 0.

    model.train()
    for x, y_gt in train_loader:
        mini_batch += 1
        x, y_gt = x.to(DEVICE), y_gt.to(DEVICE)

        # 순전파 및 손실 계산 (SUBDIVISIONS로 나눠서 스케일링)
        y_pred = model(x)
        loss = criterion(y_pred, y_gt) / SUBDIVISIONS
        loss.backward()  # 그래디언트 누적 (zero_grad 하지 않음)

        if mini_batch == SUBDIVISIONS:
            # SUBDIVISIONS번 누적 후 가중치 업데이트
            optimizer.step()
            optimizer.zero_grad()  # 그래디언트 초기화
            scheduler.step()       # 학습률 스케줄 업데이트
            mini_batch = 0         # 카운터 리셋

        av_loss += loss.item() * SUBDIVISIONS  # 원래 스케일로 복원

    av_loss /= len(train_loader)
    return av_loss, mini_batch


# ============================================================
# validate_epoch: 한 에폭 동안 검증(테스트) 수행
# ============================================================
def validate_epoch(val_loader, model, criterion):
    """
    그래디언트 계산 없이 테스트셋 손실 측정.
    VOC test set을 검증셋으로 사용.
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


# ============================================================
# train: 전체 학습 루프 (MAX_EPOCHS 에폭)
# ============================================================
def train(train_loader, test_loader, model, optimizer, criterion, scheduler,
          epoch, mini_batch, train_loss_history, test_loss_history):
    """
    tqdm 진행 바를 표시하며 MAX_EPOCHS까지 학습.
    CHECKPOINT_T 에폭마다 체크포인트 저장 (학습 재개 가능).
    학습 완료 후 최종 모델 가중치 저장.
    """
    pbar = tqdm(total=MAX_EPOCHS, desc='Training Epoch', initial=epoch, unit='epoch', position=0, leave=True)

    # 학습 재개 시 그래디언트 초기화 여부 결정
    if mini_batch == 0:
        optimizer.zero_grad()

    while epoch < MAX_EPOCHS:
        epoch += 1

        # 한 에폭 학습
        train_loss, mini_batch = train_epoch(train_loader, model, optimizer, criterion, scheduler, mini_batch)
        # 테스트셋 검증
        test_loss = validate_epoch(test_loader, model, criterion)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        # 주기적 체크포인트 저장
        if epoch % CHECKPOINT_T == 0:
            th.save({
                'epoch': epoch,
                'mini_batch': mini_batch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss_history': train_loss_history,
                'test_loss_history': test_loss_history,
                'grads': {p[0]: p[1].grad for p in model.named_parameters()}  # 그래디언트도 저장
            }, TRAINING_CHECKPOINT_PATH)

        pbar.set_postfix_str(f'Train Loss={train_loss:.3f}, Test Loss={test_loss:.3f}')
        pbar.update(1)

    # 학습 완료 후 최종 가중치 저장
    th.save(model.state_dict(), TRAINED_MODEL_WEIGHTS)
    pbar.close()


# ============================================================
# setup_train: 모델, 옵티마이저, 데이터로더 등 모든 구성 요소 초기화
# ============================================================
def setup_train():
    # ── 모델 생성 ────────────────────────────────────────────────
    model = YOLOv1(S=S, B=B, C=VOC_Detection.C).to(DEVICE)

    # ── SGD 옵티마이저 ───────────────────────────────────────────
    # 초기 학습률: Burn-in 시작점 (매우 작은 값)
    optimizer = opt.SGD(
        params=model.parameters(),
        lr=INIT_LR * (1 / BURN_IN) ** BURN_IN_POW,  # Burn-in 첫 스텝 학습률
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # ── 커스텀 학습률 스케줄러 ────────────────────────────────────
    scheduler = MultiStepScaleLR(
        optimizer, init_lr=INIT_LR,
        lr_schedule=LR_SCHEDULE,
        burn_in=BURN_IN, burn_in_pow=BURN_IN_POW
    )

    # ── YOLO 손실 함수 ────────────────────────────────────────────
    criterion = YOLO_Loss(S=S, C=VOC_Detection.C, B=B, D=D,
                          L_coord=L_COORD, L_noobj=L_NOOBJ).to(DEVICE)

    # ── 학습 데이터셋 (데이터 증강 포함) ─────────────────────────
    train_dataset = VOC_Detection(
        root_dir=PASCAL_VOC_DIR_PATH, split='train',
        transforms=transforms.Compose([
            RandomScaleTranslate(output_size=D, jitter=JITTER,
                                 resize_p=RESIZE_PROB, zoom_out_p=ZOOM_OUT_PROB, zoom_in_p=ZOOM_IN_PROB),
            RandomColorJitter(hue=HUE, sat=SATURATION, exp=EXPOSURE),
            RandomHorizontalFlip(p=0.5),
            ToYOLOTensor(S=S, C=VOC_Detection.C,
                         normalize=[[0.4549, 0.4341, 0.4010], [0.2703, 0.2672, 0.2808]])
        ])
    )

    # ── 테스트 데이터셋 (증강 없이 리사이즈만) ───────────────────
    test_dataset = VOC_Detection(
        root_dir=PASCAL_VOC_DIR_PATH, split='test',
        transforms=transforms.Compose([
            Resize(output_size=D),
            ToYOLOTensor(S=S, C=VOC_Detection.C,
                         normalize=[[0.4549, 0.4341, 0.4010], [0.2703, 0.2672, 0.2808]])
        ])
    )

    # ── DataLoader 생성 ───────────────────────────────────────────
    # 실제 미니배치 크기 = BATCH // SUBDIVISIONS (그래디언트 누적 방식 사용)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH // SUBDIVISIONS,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        shuffle=SHUFFLE, drop_last=DROP_LAST
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH // SUBDIVISIONS,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    return train_loader, test_loader, model, optimizer, scheduler, criterion


# ============================================================
# init_train: LOAD_MODEL 설정에 따라 모델 가중치 및 학습 상태 초기화
# ============================================================
def init_train(model, optimizer, scheduler):
    """
    반환: (시작 에폭, 시작 미니배치 카운터, 학습 손실 히스토리, 테스트 손실 히스토리)
    """
    if LOAD_MODEL is None:
        # 처음부터 학습 (랜덤 초기화)
        epoch, mini_batch = 0, 0
        train_loss_history, test_loss_history = [], []

    elif LOAD_MODEL == 'pretrain':
        # ImageNet 사전학습 backbone 가중치 로드 후 detection 헤드는 랜덤 초기화
        # strict=False: detection 헤드 파라미터가 없어도 오류 없이 로드
        pretrained_model_weights = th.load(PRETRAINED_MODEL_WEIGHTS)
        model.load_state_dict(pretrained_model_weights, strict=False)
        epoch, mini_batch = 0, 0
        train_loss_history, test_loss_history = [], []

    elif LOAD_MODEL == 'train':
        # 이전 학습 체크포인트에서 재개 (에폭, 옵티마이저, 스케줄러 상태 모두 복원)
        checkpoint = th.load(TRAINING_CHECKPOINT_PATH)
        epoch           = checkpoint['epoch']
        mini_batch      = checkpoint['mini_batch']
        train_loss_history = checkpoint['train_loss_history']
        test_loss_history  = checkpoint['test_loss_history']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # 누적 중이던 그래디언트도 복원 (이어서 그래디언트 누적 가능)
        for p in model.named_parameters():
            p[1].grad = checkpoint['grads'][p[0]]
    else:
        assert 0, f"LOAD_MODEL 값이 유효하지 않음: {LOAD_MODEL}"

    return epoch, mini_batch, train_loss_history, test_loss_history


# ============================================================
# main: 학습 진입점
# ============================================================
def main():
    # 1. 모든 구성 요소 초기화
    train_loader, test_loader, model, optimizer, scheduler, criterion = setup_train()
    # 2. 체크포인트/사전학습 가중치로 상태 초기화
    epoch, mini_batch, train_loss_hist, test_loss_hist = init_train(model, optimizer, scheduler)
    # 3. 학습 시작
    train(train_loader, test_loader, model, optimizer, criterion, scheduler,
          epoch, mini_batch, train_loss_hist, test_loss_hist)


if __name__ == '__main__':
    main()
