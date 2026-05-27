"""
CIFAR-10 데이터 로더.

torchvision의 CIFAR10 데이터셋을 자동 다운로드 (./data 폴더)하고,
학습용/테스트용 transform을 분리해 DataLoader로 감싸 반환.
"""

import torch
import torchvision
import torchvision.transforms as transforms


def prepare_data(batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
    """
    CIFAR-10 trainloader, testloader, classes 반환.

    Args:
        batch_size: 미니배치 크기
        num_workers: DataLoader가 사용할 워커 프로세스 수
        train_sample_size: 학습셋에서 무작위로 N개만 샘플링 (None이면 전체)
        test_sample_size: 테스트셋도 마찬가지
    """
    # ------ 학습용 transform: augmentation 포함 ------
    train_transform = transforms.Compose(
        [transforms.ToTensor(),                                      # PIL → Tensor, [0,1]로 스케일
        transforms.Resize((32, 32)),                                 # 32x32로 리사이즈 (CIFAR-10은 이미 32x32지만 안전장치)
        transforms.RandomHorizontalFlip(p=0.5),                      # 좌우 반전 50% 확률
        transforms.RandomResizedCrop(                                # 무작위 크롭 후 32x32로 리사이즈
            (32, 32),
            scale=(0.8, 1.0),                                        # 원본의 80~100% 영역 크롭
            ratio=(0.75, 1.3333333333333333),                        # 종횡비 변동 허용
            interpolation=2),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])     # [0,1] → [-1,1] (mean=0.5, std=0.5)

    # 학습 데이터셋 다운로드/로드
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    # 디버깅용: 일부 샘플로만 학습하고 싶을 때
    if train_sample_size is not None:
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)

    # DataLoader로 감싸기 (shuffle=True로 매 에폭마다 순서 섞음)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    # ------ 테스트용 transform: augmentation 없음, 정규화만 ------
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    if test_sample_size is not None:
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)

    # 테스트는 shuffle=False (재현성)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    # CIFAR-10 10개 클래스 이름
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes
