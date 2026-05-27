# Import libraries
import torch
import torchvision
import torchvision.transforms as transforms


def prepare_data(batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
    # 학습 데이터용 전처리 파이프라인 (데이터 증강 포함)
    train_transform = transforms.Compose(
        [transforms.ToTensor(), # 이미지를 PyTorch 텐서로 변환
        transforms.Resize((32, 32)), # 이미지 크기를 32x32로 조정
        transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 좌우 반전
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2), # 무작위로 자르고 크기 조정
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 픽셀 값을 [-1, 1] 범위로 정규화

    # 학습용 CIFAR-10 데이터셋 다운로드 및 로드
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    
    # 지정된 샘플 크기가 있으면 데이터셋의 일부만 무작위로 추출
    if train_sample_size is not None:
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)
    
    # 학습용 데이터로더 생성 (배치 단위로 데이터를 섞어서 공급)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    
    # 테스트 데이터용 전처리 파이프라인 (데이터 증강 없이 텐서 변환 및 정규화만 수행)
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 테스트용 CIFAR-10 데이터셋 다운로드 및 로드
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    
    if test_sample_size is not None:
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)
    
    # 테스트용 데이터로더 생성 (순서를 섞지 않음)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    # CIFAR-10의 10가지 클래스 라벨 정의
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes