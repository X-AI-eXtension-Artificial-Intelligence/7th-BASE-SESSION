import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

from vit import ViTForClassfication


def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="experiments"):
    # 학습 완료 후 설정, 지표, 모델을 한 폴더에 저장
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    
    # 설정값 저장 (config.json)
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
    
    # 학습 손실 및 정확도 지표 저장 (metrics.json)
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
    # 특정 에포크의 모델 가중치(state_dict)를 .pt 파일로 저장
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)


def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    # 저장된 실험 데이터(설정, 지표, 모델 가중치)를 불러와 복원
    outdir = os.path.join(base_dir, experiment_name)
    
    with open(os.path.join(outdir, 'config.json'), 'r') as f:
        config = json.load(f)
        
    with open(os.path.join(outdir, 'metrics.json'), 'r') as f:
        data = json.load(f)
        
    train_losses, test_losses, accuracies = data['train_losses'], data['test_losses'], data['accuracies']
    
    # 복원한 설정으로 모델 뼈대 생성 후 저장된 가중치 입히기
    model = ViTForClassfication(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))
    
    return config, model, train_losses, test_losses, accuracies


def visualize_images():
    # CIFAR-10 학습 데이터 중 30개를 무작위로 뽑아 그리드 형태로 시각화
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
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
    모델이 이미지를 판단할 때 어느 부분을 집중해서(Attention) 보는지 시각화
    """
    model.eval()
    num_images = 30
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 무작위 이미지 추출 및 전처리
    indices = torch.randperm(len(testset))[:num_images]
    raw_images = [np.asarray(testset[i][0]) for i in indices]
    labels = [testset[i][1] for i in indices]
    
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((32, 32)), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    images = torch.stack([test_transform(image) for image in raw_images]).to(device)
    model = model.to(device)
    
    # 모델 추론 및 어텐션 맵 추출
    logits, attention_maps = model(images, output_attentions=True)
    predictions = torch.argmax(logits, dim=1)
    
    # 여러 트랜스포머 블록의 어텐션 맵을 하나로 합침
    attention_maps = torch.cat(attention_maps, dim=1)
    
    # 분류를 위한 [CLS] 토큰이 다른 패치들에 부여한 어텐션 가중치만 추출
    attention_maps = attention_maps[:, :, 0, 1:]
    # 모든 어텐션 헤드의 가중치를 평균 냄
    attention_maps = attention_maps.mean(dim=1)
    
    # 1D 형태의 어텐션 배열을 2D 맵 형태(정사각형)로 변환
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)
    
    # 원래 이미지 크기(32x32)에 맞게 어텐션 맵을 부드럽게 보간(확대)
    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = F.interpolate(attention_maps, size=(32, 32), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)
    
    # 원본 이미지와 어텐션 맵이 덧씌워진 이미지를 나란히 출력
    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)
        
        # 어텐션 맵 오버레이 적용 (열화상 느낌의 'jet' 컬러맵 사용)
        extended_attention_map = np.concatenate((np.zeros((32, 32)), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        
        # 실제 정답(gt)과 예측(pred)을 비교하여 제목 표시 (맞으면 초록, 틀리면 빨강)
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
        
    if output is not None:
        plt.savefig(output)
    plt.show()