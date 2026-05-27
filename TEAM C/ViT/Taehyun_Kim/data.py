import torch
import torchvision
import torchvision.transforms as transforms


# CIFAR-10 데이터셋 로드 및 전처리
# 학습/테스트 DataLoader와 클래스 이름 반환
def prepare_data(batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):

    # -------------------------
    # 학습 데이터 전처리 파이프라인
    # -------------------------
    train_transform = transforms.Compose([
        transforms.ToTensor(),                  # PIL 이미지 → [0,1] 범위 텐서로 변환
        transforms.Resize((32, 32)),            # 32x32로 리사이즈 (CIFAR-10은 이미 32x32라 보험용)
        transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 좌우 반전 → 데이터 증강
        transforms.RandomResizedCrop(           # 랜덤 크롭 후 리사이즈 → 위치 불변성 학습
            (32, 32),
            scale=(0.8, 1.0),                   # 원본의 80~100% 크기로 크롭
            ratio=(0.75, 1.3333333333333333),    # 가로세로 비율 범위
            interpolation=2                     # 보간 방법 (BILINEAR)
        ),
        transforms.Normalize(                   # 각 채널을 평균 0.5, 표준편차 0.5로 정규화
            (0.5, 0.5, 0.5),                    # → 픽셀값을 [-1, 1] 범위로 스케일링
            (0.5, 0.5, 0.5)
        )
    ])

    # CIFAR-10 학습셋 로드 (없으면 자동 다운로드, ~170MB)
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )

    # 전체 학습 데이터 중 일부만 쓰고 싶을 때 (빠른 실험용)
    if train_sample_size is not None:
        indices = torch.randperm(len(trainset))[:train_sample_size]  # 랜덤 샘플링
        trainset = torch.utils.data.Subset(trainset, indices)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,  # 한 번에 모델에 넣을 이미지 수
        shuffle=True,           # 매 에폭마다 순서 섞음 → 일반화 성능 향상
        num_workers=num_workers # 데이터 로딩 병렬 프로세스 수
    )

    # -------------------------
    # 테스트 데이터 전처리 파이프라인
    # 증강 없이 정규화만 적용 → 공정한 성능 평가
    # -------------------------
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    # 테스트 데이터 일부만 사용할 때
    if test_sample_size is not None:
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,          # 테스트는 순서 고정 (결과 재현성 확보)
        num_workers=num_workers
    )

    # CIFAR-10의 10개 클래스 (인덱스 순서대로)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes
