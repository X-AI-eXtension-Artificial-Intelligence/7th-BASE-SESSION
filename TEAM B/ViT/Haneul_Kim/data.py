# 필요한 라이브러리 불러오기
import torch
import torchvision
import torchvision.transforms as transforms


def prepare_data(batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
    """
    CIFAR-10 데이터셋을 학습용/테스트용 DataLoader 형태로 준비하는 함수

    batch_size: 한 번에 모델에 넣을 이미지 개수
    num_workers: 데이터를 불러올 때 사용할 프로세스 개수
    train_sample_size: 학습 데이터 일부만 사용할 때 지정하는 샘플 수
    test_sample_size: 테스트 데이터 일부만 사용할 때 지정하는 샘플 수
    """

    # 학습 데이터에 적용할 전처리 정의
    # 이미지를 텐서로 바꾸고, 크기를 32x32로 맞춘 뒤, 데이터 증강과 정규화를 적용
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR-10 학습 데이터셋 다운로드 및 로드
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)

    # train_sample_size가 지정된 경우, 전체 학습 데이터 중 일부만 랜덤으로 사용
    if train_sample_size is not None:
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)

    # 학습 데이터를 배치 단위로 불러오는 DataLoader 생성
    # 학습 데이터는 매 epoch마다 순서를 섞어서 사용
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    # 테스트 데이터에 적용할 전처리 정의
    # 테스트 데이터는 평가용이므로 랜덤 증강은 하지 않고 기본 전처리만 적용
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR-10 테스트 데이터셋 다운로드 및 로드
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)

    # test_sample_size가 지정된 경우, 전체 테스트 데이터 중 일부만 랜덤으로 사용
    if test_sample_size is not None:
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)

    # 테스트 데이터를 배치 단위로 불러오는 DataLoader 생성
    # 평가는 순서가 중요하지 않으므로 shuffle=False로 설정
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    # CIFAR-10 클래스 이름 목록
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 학습 DataLoader, 테스트 DataLoader, 클래스 이름 반환
    return trainloader, testloader, classes
