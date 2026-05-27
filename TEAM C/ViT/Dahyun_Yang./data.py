# 데이터셋 준비 파일입니다. CIFAR-10을 다운로드하고 학습/테스트 DataLoader를 만듭니다.
# Import libraries
import torch
import torchvision
import torchvision.transforms as transforms


    # batch_size: 한 번에 모델에 넣을 이미지 수, num_workers: 데이터 로딩 프로세스 수입니다.
    # *_sample_size를 주면 전체 데이터 대신 일부 샘플만 사용해 빠른 실험이 가능합니다.
def prepare_data(batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
    # 학습 데이터에는 augmentation을 적용해 모델이 다양한 변형에 견디도록 합니다.
    train_transform = transforms.Compose(
        # PIL 이미지를 PyTorch 텐서로 바꾸고 값 범위를 0~1로 변환합니다.
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        # 절반 확률로 좌우 반전하여 데이터 다양성을 늘립니다.
        transforms.RandomHorizontalFlip(p=0.5),
        # 이미지를 무작위 크롭/리사이즈하여 위치와 스케일 변화에 강하게 만듭니다.
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        # 채널별 평균/표준편차로 정규화합니다. 여기서는 대략 [-1, 1] 범위로 맞춥니다.
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR-10 학습 데이터를 ./data 아래에 다운로드하고 transform을 적용합니다.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    if train_sample_size is not None:
        # Randomly sample a subset of the training set
        # randperm으로 무작위 인덱스를 만든 뒤 필요한 개수만 선택합니다.
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)
    


    # DataLoader는 Dataset을 batch 단위로 묶어 학습 루프에 공급합니다.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            # 학습 데이터는 매 epoch마다 순서를 섞어 일반화에 도움을 줍니다.
                                            shuffle=True, num_workers=num_workers)
    
    # 테스트 데이터에는 무작위 augmentation을 넣지 않고, 평가가 항상 동일하게 나오도록 합니다.
    test_transform = transforms.Compose(
        # PIL 이미지를 PyTorch 텐서로 바꾸고 값 범위를 0~1로 변환합니다.
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        # 채널별 평균/표준편차로 정규화합니다. 여기서는 대략 [-1, 1] 범위로 맞춥니다.
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR-10 테스트 데이터를 불러옵니다. train=False가 테스트 split을 의미합니다.
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)
    
    # 테스트는 성능 측정용이므로 shuffle=False로 고정된 순서를 사용합니다.
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    # CIFAR-10의 클래스 인덱스를 사람이 읽을 수 있는 이름으로 매핑합니다.
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 학습/테스트 로더와 클래스 이름을 호출자에게 반환합니다.
    return trainloader, testloader, classes
