import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

from vit import ViTForClassfication


# ============================================================
# 실험 결과 전체 저장
# config, metrics(손실/정확도), 모델 가중치를 한 번에 저장
# ============================================================
def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)  # 폴더 없으면 생성

    # 모델 설정 저장
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    # 학습 지표 저장 (에폭별 손실, 정확도)
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    # 최종 모델 가중치 저장 (파일명: model_final.pt)
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


# ============================================================
# 모델 체크포인트 저장
# 중간 에폭 저장 시에도 사용 (model_{epoch}.pt)
# ============================================================
def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    # state_dict: 모델의 학습 가능한 파라미터만 저장 (모델 구조는 제외)
    torch.save(model.state_dict(), cpfile)


# ============================================================
# 저장된 실험 로드
# config, metrics, 모델 가중치를 한 번에 복원
# ============================================================
def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)

    with open(os.path.join(outdir, 'config.json'), 'r') as f:
        config = json.load(f)

    with open(os.path.join(outdir, 'metrics.json'), 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses  = data['test_losses']
    accuracies   = data['accuracies']

    # config로 빈 모델 생성 후 저장된 가중치를 덮어씌움
    model = ViTForClassfication(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))

    return config, model, train_losses, test_losses, accuracies


# ============================================================
# CIFAR-10 샘플 이미지 시각화
# 학습셋에서 랜덤 30장을 뽑아 클래스명과 함께 표시
# ============================================================
def visualize_images():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    indices = torch.randperm(len(trainset))[:30]  # 랜덤 30개 인덱스 샘플링
    images = [np.asarray(trainset[i][0]) for i in indices]
    labels = [trainset[i][1] for i in indices]

    fig = plt.figure(figsize=(10, 10))
    for i in range(30):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]])


# ============================================================
# 어텐션 맵 시각화
# 테스트 이미지 30장에 대해 모델이 어느 부분에 집중했는지 표시
# 왼쪽: 원본 이미지 / 오른쪽: 어텐션 맵 오버레이
# ============================================================
@torch.no_grad()  # 추론 전용, 기울기 계산 불필요
def visualize_attention(model, output=None, device="cuda"):
    model.eval()
    num_images = 30

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    indices = torch.randperm(len(testset))[:num_images]
    raw_images = [np.asarray(testset[i][0]) for i in indices]  # 원본 PIL 이미지 (시각화용)
    labels = [testset[i][1] for i in indices]

    # 모델 입력용 전처리 (정규화 포함)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    images = torch.stack([test_transform(image) for image in raw_images])
    images = images.to(device)
    model = model.to(device)

    # output_attentions=True: 모든 레이어의 어텐션 맵을 함께 반환
    logits, attention_maps = model(images, output_attentions=True)
    predictions = torch.argmax(logits, dim=1)

    # 모든 레이어의 어텐션 맵을 헤드 차원으로 concat
    # 결과 shape: (batch, num_layers * num_heads, seq_len, seq_len)
    attention_maps = torch.cat(attention_maps, dim=1)

    # [CLS] 토큰(인덱스 0)이 각 패치(인덱스 1~)에 얼마나 집중하는지만 추출
    # → 분류에 실제로 사용된 어텐션 패턴을 시각화하는 것
    attention_maps = attention_maps[:, :, 0, 1:]  # (batch, heads, num_patches)

    # 헤드 평균: 헤드별 어텐션을 하나의 맵으로 합침
    attention_maps = attention_maps.mean(dim=1)   # (batch, num_patches)

    # 패치 배열을 정사각형으로 reshape: (batch, sqrt(num_patches), sqrt(num_patches))
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))  # ex) 64개 패치 → 8x8
    attention_maps = attention_maps.view(-1, size, size)

    # 어텐션 맵을 원본 이미지 크기(32x32)로 업샘플링
    attention_maps = attention_maps.unsqueeze(1)  # 채널 차원 추가 (보간 함수 요구사항)
    attention_maps = F.interpolate(attention_maps, size=(32, 32), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)    # 채널 차원 제거

    # 시각화: 이미지 왼쪽=원본, 오른쪽=어텐션 오버레이
    fig = plt.figure(figsize=(20, 10))
    # 마스크: 왼쪽(1=보임)은 원본만, 오른쪽(0=투명)은 어텐션 맵만 표시
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)

    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])

        # 이미지를 가로로 2장 붙여서 좌우 비교 레이아웃 만듦
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)

        # 오른쪽 절반에만 어텐션 맵 오버레이 (왼쪽은 마스킹)
        extended_attention_map = np.concatenate((np.zeros((32, 32)), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask == 1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')  # jet: 파랑(낮음) ~ 빨강(높음)

        # 정답과 예측이 다르면 빨간색 제목으로 표시
        gt   = classes[labels[i]]
        pred = classes[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt == pred else "red"))

    if output is not None:
        plt.savefig(output)  # 파일 경로 지정 시 저장
    plt.show()
