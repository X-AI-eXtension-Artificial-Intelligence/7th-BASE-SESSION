# 실험 저장/불러오기와 이미지·attention 시각화를 위한 유틸리티 함수 모음입니다.
import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

from vit import ViTForClassfication


    # 하나의 실험 결과를 폴더로 묶어 저장합니다: config, metrics, 최종 모델 가중치.
def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="experiments"):
    # experiments/{experiment_name} 형태의 저장 경로를 만듭니다.
    outdir = os.path.join(base_dir, experiment_name)
    # 폴더가 없으면 생성하고, 이미 있어도 에러가 나지 않게 합니다.
    os.makedirs(outdir, exist_ok=True)
    
    # Save the config
    # 모델 구조와 하이퍼파라미터를 JSON으로 저장합니다.
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
    
    # Save the metrics
    # epoch별 loss/accuracy 기록을 저장합니다.
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)
    
    # Save the model
    # 최종 학습된 모델 파라미터를 model_final.pt로 저장합니다.
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


    # 특정 epoch의 모델 state_dict를 파일로 저장합니다.
def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    # experiments/{experiment_name} 형태의 저장 경로를 만듭니다.
    outdir = os.path.join(base_dir, experiment_name)
    # 폴더가 없으면 생성하고, 이미 있어도 에러가 나지 않게 합니다.
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    # state_dict는 모델 구조가 아니라 학습된 파라미터 값만 담습니다.
    torch.save(model.state_dict(), cpfile)


    # 저장된 실험 폴더에서 config, metrics, 모델 가중치를 다시 불러옵니다.
def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    # experiments/{experiment_name} 형태의 저장 경로를 만듭니다.
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    # 모델 구조와 하이퍼파라미터를 JSON으로 저장합니다.
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    # epoch별 loss/accuracy 기록을 저장합니다.
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    # Load the model
    # 저장된 config로 동일한 모델 구조를 먼저 생성합니다.
    model = ViTForClassfication(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    # checkpoint의 파라미터를 새 모델 객체에 주입합니다.
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies


    # CIFAR-10 학습 이미지 일부를 무작위로 뽑아 그리드로 보여주는 함수입니다.
def visualize_images():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Pick 30 samples randomly
    # 전체 학습셋 중 30장을 랜덤 선택합니다.
    indices = torch.randperm(len(trainset))[:30]
    images = [np.asarray(trainset[i][0]) for i in indices]
    labels = [trainset[i][1] for i in indices]
    # Visualize the images using matplotlib
    fig = plt.figure(figsize=(10, 10))
    for i in range(30):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        # 선택한 이미지를 matplotlib subplot에 표시합니다.
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]])


# 시각화에서는 학습이 필요 없으므로 gradient 계산을 끄고 메모리 사용량을 줄입니다.
@torch.no_grad()
    # 모델이 CLS 토큰 기준으로 어떤 패치에 주목했는지 heatmap으로 시각화합니다.
def visualize_attention(model, output=None, device="cuda"):
    """
    Visualize the attention maps of the first 4 images.
    """
    # 평가 모드로 전환해 dropout 등이 꺼진 상태에서 attention을 확인합니다.
    model.eval()
    # Load random images
    num_images = 30
    # attention 확인용으로 테스트 이미지를 불러옵니다.
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Pick 30 samples randomly
    indices = torch.randperm(len(testset))[:num_images]
    raw_images = [np.asarray(testset[i][0]) for i in indices]
    labels = [testset[i][1] for i in indices]
    # Convert the images to tensors
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # PIL/NumPy 이미지를 모델 입력 텐서 batch로 변환합니다.
    images = torch.stack([test_transform(image) for image in raw_images])
    # Move the images to the device
    images = images.to(device)
    model = model.to(device)
    # Get the attention maps from the last block
    # output_attentions=True로 각 Transformer 블록의 attention 확률도 함께 받습니다.
    logits, attention_maps = model(images, output_attentions=True)
    # Get the predictions
    predictions = torch.argmax(logits, dim=1)
    # Concatenate the attention maps from all blocks
    # 여러 블록의 attention map을 head 차원 방향으로 이어 붙입니다.
    attention_maps = torch.cat(attention_maps, dim=1)
    # select only the attention maps of the CLS token
    # CLS 토큰이 각 이미지 패치에 주는 attention만 선택합니다. 0번 CLS 자신은 제외합니다.
    attention_maps = attention_maps[:, :, 0, 1:]
    # Then average the attention maps of the CLS token over all the heads
    # 모든 layer/head attention을 평균내 하나의 중요도 맵으로 요약합니다.
    attention_maps = attention_maps.mean(dim=1)
    # Reshape the attention maps to a square
    num_patches = attention_maps.size(-1)
    # 패치 개수는 정사각형이라고 가정하고 한 변의 패치 수를 계산합니다.
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)
    # Resize the map to the size of the image
    attention_maps = attention_maps.unsqueeze(1)
    # 패치 단위 heatmap을 원본 이미지 크기인 32x32로 확대합니다.
    attention_maps = F.interpolate(attention_maps, size=(32, 32), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)
    # Plot the images and the attention maps
    fig = plt.figure(figsize=(20, 10))
    # 왼쪽은 원본, 오른쪽은 attention overlay로 보여주기 위한 마스크입니다.
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)
        # Mask out the attention map of the left image
        # 오른쪽 이미지 영역에만 attention heatmap이 보이도록 2배 폭으로 확장합니다.
        extended_attention_map = np.concatenate((np.zeros((32, 32)), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        # Show the ground truth and the prediction
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        # 정답과 예측이 같으면 초록색, 다르면 빨간색 제목으로 표시합니다.
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
    if output is not None:
        # output 경로가 주어지면 화면 표시와 별개로 이미지 파일도 저장합니다.
        plt.savefig(output)
    plt.show()
