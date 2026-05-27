"""
ViT 학습 스크립트.

사용 예:
    python train.py --exp-name vit-quick --epochs 30 --batch-size 256
    python train.py --exp-name vit-full --epochs 100 --batch-size 256 --lr 1e-2

결과는 experiments/<exp-name>/ 아래에 저장됨 (config.json, metrics.json, model_final.pt).
"""

import torch
from torch import nn, optim

from utils import save_experiment, save_checkpoint
from data import prepare_data
from vit import ViTForClassfication


# ============================================================================
# 모델 config: 하이퍼파라미터를 한 곳에 모아둠
# CIFAR-10용 작은 ViT (논문 ViT-Base 대비 한참 작음, 학습 데모용)
# ============================================================================
config = {
    "patch_size": 4,                        # 패치 한 변 (32x32 이미지 → 8x8=64개 패치)
    "hidden_size": 48,                      # 임베딩/Transformer 차원 D
    "num_hidden_layers": 4,                 # Transformer 블록 개수 L (논문 Base는 12)
    "num_attention_heads": 4,               # multi-head attention의 헤드 수
    "intermediate_size": 4 * 48,            # FFN 중간 차원 (관례적으로 hidden×4)
    "hidden_dropout_prob": 0.0,             # 일반 dropout 비율
    "attention_probs_dropout_prob": 0.0,    # attention map에 적용하는 dropout
    "initializer_range": 0.02,              # 가중치 초기화 std
    "image_size": 32,                       # 입력 이미지 한 변 (CIFAR-10은 32x32)
    "num_classes": 10,                      # CIFAR-10 클래스 수
    "num_channels": 3,                      # RGB
    "qkv_bias": True,                       # Q/K/V projection에 bias 사용 여부
    "use_faster_attention": True,           # True면 최적화된 MHA, False면 직관 버전 (학습용)
}
# 안전장치: 설정 잘못 잡으면 fail-fast
assert config["hidden_size"] % config["num_attention_heads"] == 0  # 헤드별 차원이 정수여야
assert config['intermediate_size'] == 4 * config['hidden_size']    # FFN 비율 관례
assert config['image_size'] % config['patch_size'] == 0            # 이미지가 패치로 균등 분할되어야


class Trainer:
    """
    학습 루프를 담당하는 단순 trainer.
    train(): 전체 에폭 학습 + 매 에폭마다 평가
    train_epoch(): 한 에폭만 학습
    evaluate(): 테스트셋에서 정확도/loss 계산
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)      # 모델을 GPU/CPU로 이동
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        지정된 에폭 수만큼 학습 + 매 에폭 평가 + 메트릭/모델 저장.
        """
        # 에폭별 로깅용 리스트
        train_losses, test_losses, accuracies = [], [], []
        for i in range(epochs):
            # 한 에폭 학습 → 한 번 평가
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            # 메트릭 누적
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            # 주기적 체크포인트 저장 (마지막 에폭은 어차피 save_experiment에서 저장됨)
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                save_checkpoint(self.exp_name, self.model, i+1)
        # 학습 종료 후 config + 메트릭 + 최종 모델 저장
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """
        한 에폭만 학습. 표준 PyTorch 학습 루프.
        """
        self.model.train()  # dropout/BN 등을 학습 모드로 전환
        total_loss = 0
        for batch in trainloader:
            # 배치를 device로 이동 (images, labels는 모두 tensor)
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # 1) 이전 step의 gradient 초기화
            self.optimizer.zero_grad()
            # 2) Forward pass + loss 계산
            #    model(images) -> (logits, attention_maps), 학습에선 logits만 필요
            loss = self.loss_fn(self.model(images)[0], labels)
            # 3) Backward pass (gradient 계산)
            loss.backward()
            # 4) Optimizer step (파라미터 업데이트)
            self.optimizer.step()
            # loss는 배치 평균이므로 합산 시 batch 크기를 곱해야 전체 평균 계산 가능
            total_loss += loss.item() * len(images)
        # 데이터셋 전체 평균 loss
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()  # gradient 계산 비활성화 (속도↑ 메모리↓)
    def evaluate(self, testloader):
        """
        테스트셋에서 정확도와 평균 loss 계산.
        """
        self.model.eval()  # dropout/BN 등을 평가 모드로 전환
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                # 예측 (attention은 사용 안 하므로 _로 받음)
                logits, _ = self.model(images)

                # loss 누적
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # 정확도 누적: argmax로 예측 클래스 뽑고 정답과 비교
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def parse_args():
    """
    커맨드라인 인자 파싱.
    --exp-name만 필수, 나머지는 기본값 있음.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)              # 실험 이름 (저장 경로)
    parser.add_argument("--batch-size", type=int, default=256)              # 미니배치 크기
    parser.add_argument("--epochs", type=int, default=100)                  # 학습 에폭 수
    parser.add_argument("--lr", type=float, default=1e-2)                   # 학습률
    parser.add_argument("--device", type=str)                               # cuda 또는 cpu
    parser.add_argument("--save-model-every", type=int, default=0)          # N 에폭마다 체크포인트 저장

    args = parser.parse_args()
    # device 미지정 시 GPU 우선 자동 선택
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main():
    args = parse_args()
    # 인자 풀기
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    save_model_every_n_epochs = args.save_model_every

    # 1) 데이터 로드 (CIFAR-10, 자동 다운로드)
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)

    # 2) 모델/옵티마이저/loss 함수 준비
    model = ViTForClassfication(config)
    # AdamW: Adam + weight decay (논문에서도 weight decay 큼직하게 0.1 사용)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # 다중 클래스 분류에 표준인 cross entropy
    loss_fn = nn.CrossEntropyLoss()

    # 3) Trainer 생성 → 학습 시작
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device=device)
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)


if __name__ == "__main__":
    main()
