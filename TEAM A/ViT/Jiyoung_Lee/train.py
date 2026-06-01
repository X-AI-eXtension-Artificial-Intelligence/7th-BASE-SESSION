import torch
from torch import nn, optim

from utils import save_experiment, save_checkpoint
from data import prepare_data
from vit import ViTForClassfication


# 모델 하이퍼파라미터 설정
# ViT 논문의 ViT-Base(12 layers, hidden 768)보다 훨씬 작은 경량 모델.
# CIFAR-10(32x32) 규모에 맞게 조정하여 빠른 학습 가능.
config = {
    "patch_size": 4,              # 32x32 이미지 → (32/4)^2 = 64개 패치
    "hidden_size": 48,            # 각 패치 벡터의 차원
    "num_hidden_layers": 4,       # Transformer Block 수
    "num_attention_heads": 4,     # Multi-Head Attention의 head 수
    "intermediate_size": 4 * 48, # MLP의 중간 차원 (논문: 4 * hidden_size)
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,            # CIFAR-10 클래스 수
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True, # FasterMultiHeadAttention 사용 여부
}

# 설정 유효성 검사
# hidden_size는 num_attention_heads로 나누어 떨어져야 한다.
# (각 head의 차원 = hidden_size / num_heads 이므로)
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0


class Trainer:
    """
    학습 루프를 관리하는 트레이너 클래스.
    모델, 옵티마이저, 손실 함수를 받아 학습/평가를 수행한다.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        지정한 epoch 수만큼 학습을 수행하고 결과를 저장.

        매 epoch마다:
          - train_epoch()으로 학습
          - evaluate()로 test set 성능 측정
          - loss 및 accuracy 기록
        """
        train_losses, test_losses, accuracies = [], [], []

        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

            # n epoch마다 중간 체크포인트 저장 (마지막 epoch 제외)
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                save_checkpoint(self.exp_name, self.model, i+1)

        # 학습 완료 후 최종 모델 및 메트릭 저장
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """
        한 epoch의 학습 수행.

        학습 루프:
          1. 배치를 device로 이동
          2. 기울기 초기화 (zero_grad)
          3. forward pass → loss 계산
          4. backward pass → 기울기 계산
          5. optimizer.step() → 가중치 업데이트
        """
        self.model.train()  # 학습 모드: Dropout 활성화
        total_loss = 0

        for batch in trainloader:
            batch = [t.to(self.device) for t in batch]
            images, labels = batch

            self.optimizer.zero_grad()

            # 모델 출력: (logits, attention_probs)
            # 학습 시에는 logits만 사용
            loss = self.loss_fn(self.model(images)[0], labels)

            loss.backward()
            self.optimizer.step()

            # 배치 크기를 곱해 샘플 단위 누적 loss 계산
            total_loss += loss.item() * len(images)

        # 전체 데이터셋 기준 평균 loss 반환
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        """
        Test set에 대한 accuracy 및 loss 계산.

        @torch.no_grad(): 평가 시 기울기 계산 불필요 → 메모리/속도 절약
        """
        self.model.eval()  # 평가 모드: Dropout 비활성화
        total_loss = 0
        correct = 0

        with torch.no_grad():
            for batch in testloader:
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                logits, _ = self.model(images)

                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # argmax로 가장 높은 확률의 클래스를 예측으로 선택
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()

        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def parse_args():
    """커맨드라인 인자 파싱. 실험 이름, 배치 크기, epoch 수, lr 등을 받는다."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-model-every", type=int, default=0)

    args = parser.parse_args()
    # device 미지정 시 GPU 있으면 cuda, 없으면 cpu 자동 선택
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main():
    args = parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    save_model_every_n_epochs = args.save_model_every

    # CIFAR-10 데이터 로딩
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)

    # 모델 생성
    model = ViTForClassfication(config)

    # AdamW 옵티마이저: Adam + weight decay (L2 정규화)
    # weight_decay=1e-2로 과적합 방지
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # CrossEntropyLoss: 다중 클래스 분류의 표준 손실 함수
    # 내부적으로 softmax + NLL loss를 합친 형태
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device=device)
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)


if __name__ == "__main__":
    main()