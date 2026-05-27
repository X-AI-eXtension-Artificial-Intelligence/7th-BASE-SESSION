# 학습 실행 파일입니다. config 설정, Trainer 클래스, CLI 인자 처리, main 루프를 포함합니다.
import torch
from torch import nn, optim

from utils import save_experiment, save_checkpoint
from data import prepare_data
from vit import ViTForClassfication


# ViT 모델과 학습에 사용할 기본 하이퍼파라미터입니다.
config = {
    # patch_size=4이면 32x32 이미지가 8x8=64개 패치로 나뉩니다.
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    # 각 패치/CLS 토큰을 표현하는 벡터 차원입니다.
    "hidden_size": 48,
    # Transformer Encoder 블록을 몇 층 쌓을지 정합니다.
    "num_hidden_layers": 4,
    # Multi-head attention의 head 개수입니다. hidden_size가 이 값으로 나누어떨어져야 합니다.
    "num_attention_heads": 4,
    # MLP 내부 확장 차원입니다. 일반적으로 hidden_size의 4배를 사용합니다.
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    # CIFAR-10은 클래스가 10개이므로 최종 출력 차원도 10입니다.
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    # True면 Q/K/V를 한 번에 계산하는 최적화된 attention 구현을 사용합니다.
    "use_faster_attention": True,
}
# 아래 assert는 잘못된 설정을 초기에 발견하기 위한 안전장치입니다.
# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0


# 모델 학습/평가/저장을 담당하는 간단한 Trainer 클래스입니다.
class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        # 모델 파라미터를 GPU 또는 CPU로 이동합니다.
        self.model = model.to(device)
        # optimizer는 gradient를 사용해 모델 파라미터를 업데이트합니다.
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        # epoch별 학습 손실, 테스트 손실, 정확도를 저장해 나중에 시각화/분석할 수 있게 합니다.
        train_losses, test_losses, accuracies = [], [], []
        # Train the model
        # 전체 데이터셋을 epochs 횟수만큼 반복 학습합니다.
        for i in range(epochs):
            # 한 epoch 동안 학습 데이터를 사용해 파라미터를 업데이트합니다.
            train_loss = self.train_epoch(trainloader)
            # 같은 epoch 끝에서 테스트셋 성능을 측정합니다.
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                # 중간 체크포인트를 저장하면 학습 중단 후 재사용하거나 비교할 수 있습니다.
                save_checkpoint(self.exp_name, self.model, i+1)
        # Save the experiment
        # 최종 모델, 설정, 성능 기록을 experiments 폴더에 저장합니다.
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        # train 모드: Dropout/BatchNorm 등이 학습 방식으로 동작합니다.
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            # Move the batch to the device
            # 이미지와 라벨 텐서를 모델과 같은 device로 옮깁니다.
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # Zero the gradients
            # PyTorch는 gradient를 누적하므로, 매 batch마다 이전 gradient를 지웁니다.
            self.optimizer.zero_grad()
            # Calculate the loss
            # model(images)[0]은 클래스별 logits이고, CrossEntropyLoss가 정답 라벨과 비교합니다.
            loss = self.loss_fn(self.model(images)[0], labels)
            # Backpropagate the loss
            # loss를 기준으로 각 파라미터의 gradient를 계산합니다.
            loss.backward()
            # Update the model's parameters
            # 계산된 gradient를 사용해 파라미터를 한 번 업데이트합니다.
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        # eval 모드: Dropout 등을 끄고 평가가 안정적으로 나오게 합니다.
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch]
                images, labels = batch
                
                # Get predictions
                # 평가에서는 attention map이 필요 없으므로 logits만 사용합니다.
                logits, _ = self.model(images)

                # Calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                # 가장 큰 logit을 가진 클래스 인덱스를 예측값으로 선택합니다.
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        # 전체 테스트 샘플 중 맞춘 비율을 정확도로 계산합니다.
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    # 실험 이름은 저장 폴더 이름으로 쓰이므로 필수 인자입니다.
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-model-every", type=int, default=0)

    args = parser.parse_args()
    # 사용자가 device를 지정하지 않으면 CUDA 가능 여부에 따라 자동 선택합니다.
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main():
    # 터미널에서 전달한 학습 옵션을 읽습니다.
    args = parse_args()
    # Training parameters
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    save_model_every_n_epochs = args.save_model_every
    # Load the CIFAR10 dataset
    # CIFAR-10 DataLoader를 준비합니다.
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)
    # Create the model, optimizer, loss function and trainer
    # config에 정의된 구조로 ViT 분류 모델을 생성합니다.
    model = ViTForClassfication(config)
    # AdamW는 Transformer 학습에 자주 쓰이며 weight_decay로 과적합을 줄입니다.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device=device)
    # 실제 학습 루프를 시작합니다.
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)


if __name__ == "__main__":
    main()
