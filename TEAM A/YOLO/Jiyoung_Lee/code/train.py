import torch as th
import torchvision.transforms as transforms
import torch.optim as opt

from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import VOC_Detection

from transforms import (
    RandomScaleTranslate,
    Resize,
    RandomColorJitter,
    RandomHorizontalFlip,
    ToYOLOTensor
)

from model import YOLOv1
from loss import YOLO_Loss

from tqdm import tqdm
from typing import List, Tuple, Dict


# =========================================
# Model Hyperparameters
# =========================================

# grid size
S = 7

# grid 당 bbox 개수
B = 2

# input image size
D = 448


# =========================================
# Loss Hyperparameters
# =========================================

# localization loss 가중치
L_COORD = 5.0

# no-object loss 가중치
L_NOOBJ = 0.5


# =========================================
# Data Augmentation Hyperparameters
# =========================================

# HSV augmentation
HUE = 0.1
SATURATION = 1.5
EXPOSURE = 1.5

# resize 확률
RESIZE_PROB = 0.2

# zoom out 확률
ZOOM_OUT_PROB = 0.4

# zoom in 확률
ZOOM_IN_PROB = 0.4

# 이미지 변형 정도
JITTER = 0.2


# =========================================
# DataLoader Hyperparameters
# =========================================

# 전체 batch size
BATCH = 64

# gradient accumulation 횟수
SUBDIVISIONS = 8

NUM_WORKERS = 10

SHUFFLE = True

PIN_MEMORY = True

DROP_LAST = True


# =========================================
# Training Hyperparameters
# =========================================

MAX_EPOCHS = 156

# 초기 learning rate
INIT_LR = 0.0005

# warmup batch 수
BURN_IN = 100

# warmup 증가 곡선 power
BURN_IN_POW = 2.


"""
(step, scale)

해당 step에서
lr *= scale 수행
"""

LR_SCHEDULE = [
    (750, 2.0),
    (1500, 2.0),
    (2250, 1.25),
    (3250, 1.60),
    (5500, 1.25),
    (15000, 0.8),
    (20000, 0.625),
    (25000, 0.8),
    (30000, 0.5),
    (35000, 0.5)
]

MOMENTUM = 0.9

WEIGHT_DECAY = 0.0005


# =========================================
# VOC Dataset 경로
# =========================================

PASCAL_VOC_DIR_PATH = "/media/soul/DATA/cv_datasets/PASCAL_VOC/VOC_Detection"


# =========================================
# Device 설정
# =========================================

DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'


# =========================================
# Checkpoint 설정
# =========================================

"""
LOAD_MODEL:
    pretrain -> pretrained weight 사용
    train -> checkpoint 이어서 학습
    None -> 처음부터 학습
"""

LOAD_MODEL = 'pretrain'


# pretrained model weight 경로
PRETRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object Detection/checkpoints/pretrained_model_weights.pt"


# training checkpoint 경로
TRAINING_CHECKPOINT_PATH = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object Detection/checkpoints/training_checkpoint.pt"


# 최종 trained weight 저장 경로
TRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object Detection/checkpoints/trained_model_weights.pt"


# checkpoint 저장 주기
CHECKPOINT_T = 10


# ============================================================
# Custom Learning Rate Scheduler
# ============================================================

class MultiStepScaleLR:
    """
    custom learning rate scheduler

    기능:
        1. warmup(burn-in)
        2. 특정 step마다 lr scaling
    """

    def __init__(
            self,
            optimizer: opt.SGD,
            init_lr: float,
            lr_schedule: List[Tuple[int, float]],
            burn_in: int,
            burn_in_pow: float
    ) -> None:

        # optimizer 저장
        self.optimizer = optimizer

        # step, scale 분리 저장
        self.steps, self.scales = zip(*lr_schedule)

        # warmup batch 수
        self.burn_in = burn_in

        # 최종 초기 learning rate
        self.init_lr = init_lr

        # warmup 증가 곡선 power
        self.pow = burn_in_pow

        # 현재 batch 수
        self.batch = 0

        # 다음 scaling step index
        self.next_step_ind = 0

    def step(self) -> None:
        """
        learning rate update
        """

        # batch 증가
        self.batch += 1

        # =====================================
        # Warmup(Burn-In)
        # =====================================

        if self.batch < self.burn_in:

            """
            learning rate를 천천히 증가

            lr =
                init_lr *
                ((batch+1)/burn_in)^pow
            """

            self.optimizer.param_groups[0]['lr'] = (
                    self.init_lr *
                    ((self.batch + 1) / self.burn_in) ** self.pow
            )

        # =====================================
        # Multi-step LR scaling
        # =====================================

        elif (
                self.next_step_ind < len(self.steps)
                and self.batch == self.steps[self.next_step_ind]
        ):

            # lr scaling
            self.optimizer.param_groups[0]['lr'] *= \
                self.scales[self.next_step_ind]

            # 다음 step으로 이동
            self.next_step_ind += 1

    def state_dict(self) -> dict:
        """
        scheduler 상태 저장
        """

        return {
            key: value
            for (key, value) in self.__dict__.items()
            if key != 'optimizer'
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        scheduler 상태 복원
        """

        self.__dict__.update(state_dict)


# ============================================================
# Train One Epoch
# ============================================================

def train_epoch(
        train_loader: DataLoader,
        model: YOLOv1,
        optimizer: opt.SGD,
        criterion: YOLO_Loss,
        scheduler: MultiStepScaleLR,
        mini_batch: int
) -> Tuple[float, int]:
    """
    한 epoch 학습
    """

    # 평균 loss 저장용
    av_loss = 0.

    # train mode
    model.train()

    for x, y_gt in train_loader:

        # accumulation counter 증가
        mini_batch += 1

        # GPU 이동
        x = x.to(DEVICE)
        y_gt = y_gt.to(DEVICE)

        # prediction
        y_pred = model(x)

        """
        gradient accumulation 위해
        loss를 subdivisions로 나눔
        """

        loss = criterion(y_pred, y_gt) / SUBDIVISIONS

        # backward
        loss.backward()

        """
        SUBDIVISIONS 번 gradient 누적 후
        optimizer step 수행
        """

        if mini_batch == SUBDIVISIONS:

            optimizer.step()

            optimizer.zero_grad()

            scheduler.step()

            mini_batch = 0

        # 평균 loss 계산용
        av_loss += loss.item() * SUBDIVISIONS

    # 평균 loss 계산
    av_loss /= len(train_loader)

    return av_loss, mini_batch


# ============================================================
# Validation
# ============================================================

def validate_epoch(
        val_loader: DataLoader,
        model: YOLOv1,
        criterion: YOLO_Loss
) -> float:
    """
    validation loss 계산
    """

    av_loss = 0.

    with th.no_grad():

        # eval mode
        model.eval()

        for x, y_gt in val_loader:

            x = x.to(DEVICE)
            y_gt = y_gt.to(DEVICE)

            # prediction
            y_pred = model(x)

            # loss 계산
            loss = criterion(y_pred, y_gt)

            av_loss += loss.item()

    # 평균 loss
    av_loss /= len(val_loader)

    return av_loss


# ============================================================
# Full Training Loop
# ============================================================

def train(
        train_loader: DataLoader,
        test_loader: DataLoader,
        model: YOLOv1,
        optimizer: opt.SGD,
        criterion: YOLO_Loss,
        scheduler: MultiStepScaleLR,
        epoch: int,
        mini_batch: int,
        train_loss_history: List[float],
        test_loss_history: List[float]
) -> None:
    """
    전체 training loop
    """

    # tqdm progress bar
    pbar = tqdm(
        total=MAX_EPOCHS,
        desc='Training Epoch',
        initial=epoch,
        unit='epoch',
        position=0,
        leave=True
    )

    # accumulation 시작 전 gradient 초기화
    if mini_batch == 0:
        optimizer.zero_grad()

    while epoch < MAX_EPOCHS:

        # epoch 증가
        epoch += 1

        # train
        train_loss, mini_batch = train_epoch(
            train_loader,
            model,
            optimizer,
            criterion,
            scheduler,
            mini_batch
        )

        # validation
        test_loss = validate_epoch(
            test_loader,
            model,
            criterion
        )

        # history 저장
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        # =====================================
        # Checkpoint 저장
        # =====================================

        if epoch % CHECKPOINT_T == 0:

            th.save({

                'epoch': epoch,

                'mini_batch': mini_batch,

                'model_state_dict': model.state_dict(),

                'optimizer_state_dict': optimizer.state_dict(),

                'scheduler_state_dict': scheduler.state_dict(),

                'train_loss_history': train_loss_history,

                'test_loss_history': test_loss_history,

                # gradient 저장
                'grads': {
                    p[0]: p[1].grad
                    for p in model.named_parameters()
                }

            }, TRAINING_CHECKPOINT_PATH)

        # tqdm 출력
        pbar.set_postfix_str(
            f'Train Loss={train_loss:.3f}, Test Loss={test_loss:.3f}'
        )

        pbar.update(1)

    # =====================================
    # 최종 모델 저장
    # =====================================

    th.save(
        model.state_dict(),
        TRAINED_MODEL_WEIGHTS
    )

    pbar.close()


# ============================================================
# Setup Training
# ============================================================

def setup_train():
    """
    model, optimizer, scheduler,
    dataset, dataloader 생성
    """

    # =====================================
    # Model 생성
    # =====================================

    model = YOLOv1(
        S=S,
        B=B,
        C=VOC_Detection.C
    ).to(DEVICE)

    # =====================================
    # Optimizer 생성
    # =====================================

    optimizer = opt.SGD(
        params=model.parameters(),

        # warmup 시작 lr
        lr=INIT_LR * (1 / BURN_IN) ** BURN_IN_POW,

        momentum=MOMENTUM,

        weight_decay=WEIGHT_DECAY
    )

    # =====================================
    # Scheduler 생성
    # =====================================

    scheduler = MultiStepScaleLR(
        optimizer,
        init_lr=INIT_LR,
        lr_schedule=LR_SCHEDULE,
        burn_in=BURN_IN,
        burn_in_pow=BURN_IN_POW
    )

    # =====================================
    # Loss 생성
    # =====================================

    criterion = YOLO_Loss(
        S=S,
        C=VOC_Detection.C,
        B=B,
        D=D,
        L_coord=L_COORD,
        L_noobj=L_NOOBJ
    ).to(DEVICE)

    # =====================================
    # Train Dataset
    # =====================================

    train_dataset = VOC_Detection(
        root_dir=PASCAL_VOC_DIR_PATH,
        split='train',

        transforms=transforms.Compose([

            RandomScaleTranslate(
                output_size=D,
                jitter=JITTER,
                resize_p=RESIZE_PROB,
                zoom_out_p=ZOOM_OUT_PROB,
                zoom_in_p=ZOOM_IN_PROB
            ),

            RandomColorJitter(
                hue=HUE,
                sat=SATURATION,
                exp=EXPOSURE
            ),

            RandomHorizontalFlip(p=0.5),

            ToYOLOTensor(
                S=S,
                C=VOC_Detection.C,

                normalize=[
                    [0.4549, 0.4341, 0.4010],
                    [0.2703, 0.2672, 0.2808]
                ]
            )
        ])
    )

    # =====================================
    # Test Dataset
    # =====================================

    test_dataset = VOC_Detection(
        root_dir=PASCAL_VOC_DIR_PATH,
        split='test',

        transforms=transforms.Compose([

            Resize(output_size=D),

            ToYOLOTensor(
                S=S,
                C=VOC_Detection.C,

                normalize=[
                    [0.4549, 0.4341, 0.4010],
                    [0.2703, 0.2672, 0.2808]
                ]
            )
        ])
    )

    # =====================================
    # Train Loader
    # =====================================

    train_loader = DataLoader(
        dataset=train_dataset,

        # accumulation 고려
        batch_size=BATCH // SUBDIVISIONS,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

        shuffle=SHUFFLE,

        drop_last=DROP_LAST
    )

    # =====================================
    # Test Loader
    # =====================================

    test_loader = DataLoader(
        dataset=test_dataset,

        batch_size=BATCH // SUBDIVISIONS,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY
    )

    return (
        train_loader,
        test_loader,
        model,
        optimizer,
        scheduler,
        criterion
    )


# ============================================================
# Initialize Training
# ============================================================

def init_train(
        model: YOLOv1,
        optimizer: opt.SGD,
        scheduler: MultiStepScaleLR
) -> Tuple[int, int, List[float], List[float]]:
    """
    checkpoint/pretrained weight 로드
    """

    # =====================================
    # 처음부터 학습
    # =====================================

    if LOAD_MODEL is None:

        epoch = 0

        mini_batch = 0

        train_loss_history = []

        test_loss_history = []

    # =====================================
    # pretrained weight 로드
    # =====================================

    elif LOAD_MODEL == 'pretrain':

        pretrained_model_weights = th.load(
            PRETRAINED_MODEL_WEIGHTS
        )

        # detection head 제외 가능
        model.load_state_dict(
            pretrained_model_weights,
            strict=False
        )

        epoch = 0

        mini_batch = 0

        train_loss_history = []

        test_loss_history = []

    # =====================================
    # checkpoint 이어서 학습
    # =====================================

    elif LOAD_MODEL == 'train':

        checkpoint = th.load(
            TRAINING_CHECKPOINT_PATH
        )

        epoch = checkpoint['epoch']

        mini_batch = checkpoint['mini_batch']

        train_loss_history = checkpoint['train_loss_history']

        test_loss_history = checkpoint['test_loss_history']

        model.load_state_dict(
            checkpoint['model_state_dict']
        )

        optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )

        scheduler.load_state_dict(
            checkpoint['scheduler_state_dict']
        )

        # gradient 복원
        for p in model.named_parameters():

            p[1].grad = checkpoint['grads'][p[0]]

    else:
        assert 0

    return (
        epoch,
        mini_batch,
        train_loss_history,
        test_loss_history
    )


# ============================================================
# Main
# ============================================================

def main():

    # training 요소 생성
    (
        train_loader,
        test_loader,
        model,
        optimizer,
        scheduler,
        criterion
    ) = setup_train()

    # checkpoint/pretrained 초기화
    (
        epoch,
        mini_batch,
        train_loss_hist,
        test_loss_hist
    ) = init_train(
        model,
        optimizer,
        scheduler
    )

    # 학습 시작
    train(
        train_loader,
        test_loader,
        model,
        optimizer,
        criterion,
        scheduler,
        epoch,
        mini_batch,
        train_loss_hist,
        test_loss_hist
    )


if __name__ == '__main__':
    main()