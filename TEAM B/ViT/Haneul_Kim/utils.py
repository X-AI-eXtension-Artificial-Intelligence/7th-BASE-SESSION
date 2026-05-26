# 실험 결과 저장, 모델 로드, 이미지/attention 시각화에 필요한 라이브러리 불러오기
import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

from vit import ViTForClassfication


def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="experiments"):
    """
    하나의 실험 결과를 폴더에 저장하는 함수

    저장되는 항목:
    1. config.json: 모델 및 학습 설정값
    2. metrics.json: epoch별 loss와 accuracy
    3. model_final.pt: 최종 학습 모델 가중치
    """
    # 실험 결과를 저장할 폴더 경로 생성
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    # 모델 설정값을 JSON 파일로 저장
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    # 학습 과정에서 기록한 loss와 accuracy를 JSON 파일로 저장
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    # 최종 모델 가중치 저장
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    """
    모델의 현재 가중치를 checkpoint 파일로 저장하는 함수
    """
    # checkpoint를 저장할 실험 폴더 생성
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    # model_epoch.pt 형태로 파일 이름 지정
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')

    # 모델 파라미터만 저장
    torch.save(model.state_dict(), cpfile)


def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    """
    저장된 실험 설정, 성능 기록, 모델 가중치를 다시 불러오는 함수
    """
    outdir = os.path.join(base_dir, experiment_name)

    # 저장된 config 파일 로드
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)

    # 저장된 metric 파일 로드
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']

    # config를 기반으로 동일한 모델 구조 생성
    model = ViTForClassfication(config)

    # 저장된 checkpoint 가중치를 모델에 적용
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))

    # 실험 설정, 모델, 성능 기록 반환
    return config, model, train_losses, test_losses, accuracies


def visualize_images():
    """
    CIFAR-10 학습 데이터에서 이미지를 랜덤으로 30장 뽑아 시각화하는 함수
    """
    # CIFAR-10 학습 데이터 로드
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 전체 데이터 중 30개 샘플을 랜덤 선택
    indices = torch.randperm(len(trainset))[:30]
    images = [np.asarray(trainset[i][0]) for i in indices]
    labels = [trainset[i][1] for i in indices]

    # matplotlib을 이용해 6행 5열 형태로 이미지 출력
    fig = plt.figure(figsize=(10, 10))
    for i in range(30):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]])


@torch.no_grad()
def visualize_attention(model, output=None, device="cuda"):
    """
    ViT 모델이 이미지의 어느 패치에 집중했는지 attention map으로 시각화하는 함수
    """
    # 모델을 평가 모드로 전환
    model.eval()

    # 시각화할 이미지 개수 설정
    num_images = 30

    # CIFAR-10 테스트 데이터 로드
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 테스트 데이터에서 30개 샘플 랜덤 선택
    indices = torch.randperm(len(testset))[:num_images]
    raw_images = [np.asarray(testset[i][0]) for i in indices]
    labels = [testset[i][1] for i in indices]

    # 모델 입력 형식에 맞게 이미지 전처리
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    images = torch.stack([test_transform(image) for image in raw_images])

    # 이미지와 모델을 지정한 장치로 이동
    images = images.to(device)
    model = model.to(device)

    # output_attentions=True로 설정하여 attention map까지 함께 반환받음
    logits, attention_maps = model(images, output_attentions=True)

    # logits에서 가장 큰 값을 가진 클래스를 예측 결과로 선택
    predictions = torch.argmax(logits, dim=1)

    # 모든 Transformer block의 attention map을 head 차원 방향으로 이어붙임
    attention_maps = torch.cat(attention_maps, dim=1)

    # CLS token이 각 이미지 패치에 주는 attention만 선택
    # 0번 token은 CLS token이고, 1번부터가 이미지 패치 token
    attention_maps = attention_maps[:, :, 0, 1:]

    # 여러 attention head의 값을 평균 내어 하나의 attention map으로 만듦
    attention_maps = attention_maps.mean(dim=1)

    # 패치 개수에 맞게 정사각형 형태로 변환
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)

    # attention map을 원본 이미지 크기인 32x32로 확대
    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = F.interpolate(attention_maps, size=(32, 32), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)

    # 원본 이미지와 attention map을 함께 시각화
    fig = plt.figure(figsize=(20, 10))

    # 왼쪽 원본 이미지에는 attention map이 덮이지 않도록 mask 생성
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)

    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])

        # 같은 이미지를 좌우로 붙임
        # 왼쪽은 원본, 오른쪽은 attention map overlay 용도
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)

        # 오른쪽 이미지에만 attention map을 표시하기 위해 왼쪽 영역은 0으로 채움
        extended_attention_map = np.concatenate((np.zeros((32, 32)), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)

        # attention map을 반투명하게 덮어서 표시
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')

        # 실제 정답과 모델 예측 결과 표시
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))

    # output 경로가 지정되어 있으면 이미지 파일로 저장
    if output is not None:
        plt.savefig(output)

    # 화면에 시각화 결과 출력
    plt.show()
