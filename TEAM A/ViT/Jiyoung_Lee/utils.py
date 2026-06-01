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
    실험 결과 전체를 저장하는 함수.

    저장 내용:
      - config.json: 모델 하이퍼파라미터
      - metrics.json: 학습/테스트 loss 및 accuracy 기록
      - model_final.pt: 최종 모델 가중치
    """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    # 모델 config를 JSON으로 저장
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    # 학습 메트릭(loss, accuracy)을 JSON으로 저장
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    # 최종 모델 체크포인트 저장
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    """
    모델 가중치를 체크포인트로 저장.
    state_dict만 저장하여 파일 크기 최소화.
    (모델 구조는 코드에서, 가중치만 파일에서 불러오는 방식)
    """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)


def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    """
    저장된 실험 결과를 불러오는 함수.

    Returns:
        config: 모델 하이퍼파라미터
        model: 가중치가 로드된 모델
        train_losses, test_losses, accuracies: 학습 기록
    """
    outdir = os.path.join(base_dir, experiment_name)

    # config 로드
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)

    # 메트릭 로드
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']

    # 모델 생성 후 저장된 가중치 로드
    model = ViTForClassfication(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))

    return config, model, train_losses, test_losses, accuracies


def visualize_images():
    """
    CIFAR-10 학습 데이터에서 30개 샘플을 랜덤으로 선택하여 시각화.
    데이터 확인용 유틸리티 함수.
    """
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 30개 샘플 랜덤 선택
    indices = torch.randperm(len(trainset))[:30]
    images = [np.asarray(trainset[i][0]) for i in indices]
    labels = [trainset[i][1] for i in indices]

    fig = plt.figure(figsize=(10, 10))
    for i in range(30):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]])


@torch.no_grad()
def visualize_attention(model, output=None, device="cuda"):
    """
    모델의 Attention map을 시각화하는 함수.

    동작 방식:
      1. 랜덤 테스트 이미지 30장 선택
      2. 모델에 output_attentions=True로 forward → 모든 레이어의 attention 반환
      3. 모든 레이어의 attention을 concat
      4. [CLS] 토큰이 각 패치에 얼마나 집중하는지 추출 (index 0 → 나머지)
      5. 모든 head의 평균을 계산하여 단일 attention map 생성
      6. 패치 수 → 이미지 크기로 interpolation하여 원본 이미지에 overlay

    @torch.no_grad(): 추론 시 기울기 불필요 → 메모리 절약
    """
    model.eval()
    num_images = 30

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 원본 PIL 이미지와 레이블 수집
    indices = torch.randperm(len(testset))[:num_images]
    raw_images = [np.asarray(testset[i][0]) for i in indices]
    labels = [testset[i][1] for i in indices]

    # 모델 입력용 전처리
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    images = torch.stack([test_transform(image) for image in raw_images])
    images = images.to(device)
    model = model.to(device)

    # output_attentions=True: 모든 Block의 attention map 반환
    logits, attention_maps = model(images, output_attentions=True)
    predictions = torch.argmax(logits, dim=1)

    # 모든 레이어의 attention map을 head 차원으로 concat
    # 각 attention_map: (batch, num_heads, seq_len, seq_len)
    # concat 후: (batch, num_layers * num_heads, seq_len, seq_len)
    attention_maps = torch.cat(attention_maps, dim=1)

    # [CLS] 토큰(index 0)이 나머지 패치들에 얼마나 집중하는지 추출
    # (batch, num_heads_total, seq_len, seq_len) → (batch, num_heads_total, num_patches)
    attention_maps = attention_maps[:, :, 0, 1:]

    # 모든 head의 평균 → (batch, num_patches)
    attention_maps = attention_maps.mean(dim=1)

    # 패치 수 → (size, size) 2D로 reshape
    # 예: 64 패치 → (8, 8)
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)

    # 패치 크기(8x8) → 이미지 크기(32x32)로 bilinear interpolation
    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = F.interpolate(attention_maps, size=(32, 32), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)

    # 시각화: 원본 이미지(왼쪽)와 attention overlay(오른쪽)를 나란히 표시
    fig = plt.figure(figsize=(20, 10))
    # 왼쪽(원본)=1, 오른쪽(attention)=0 인 마스크
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)

    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])

        # 원본 이미지를 좌우로 이어붙여 표시
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)

        # 왼쪽(원본)은 attention을 가리고, 오른쪽에만 attention map overlay
        extended_attention_map = np.concatenate((np.zeros((32, 32)), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')

        # 예측 정답이면 초록, 틀리면 빨간색으로 제목 표시
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))

    if output is not None:
        plt.savefig(output)
    plt.show()