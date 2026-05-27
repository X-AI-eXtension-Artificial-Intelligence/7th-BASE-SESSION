import torch
from torch import nn, optim

from utils import save_experiment, save_checkpoint
from data import prepare_data
from vit import ViTForClassfication


# ============================================================
# 모델 하이퍼파라미터 설정
# ============================================================
config = {
    "patch_size": 4,           # 이미지를 4x4 픽셀 단위로 패치 분할 → 32x32 이미지면 64개 패치
    "hidden_size": 48,         # 모든 레이어의 임베딩 차원 크기
    "num_hidden_layers": 4,    # 트랜스포머 블록 개수 (원논문 ViT-Base는 12개)
    "num_attention_heads": 4,  # 어텐션 헤드 수. hidden_size를 이 수로 나눠야 함 (48/4=12)
    "intermediate_size": 4 * 48,  # MLP 내부 확장 크기. 관행적으로 hidden_size의 4배
    "hidden_dropout_prob": 0.0,         # 임베딩/MLP 출력에 적용할 드롭아웃 비율
    "attention_probs_dropout_prob": 0.0,# 어텐션 가중치에 적용할 드롭아웃 비율
    "initializer_range": 0.02, # 가중치 초기화 시 정규분포의 표준편차
    "image_size": 32,          # 입력 이미지 크기 (CIFAR-10: 32x32)
    "num_classes": 10,         # 분류할 클래스 수 (CIFAR-10: 10개)
    "num_channels": 3,         # 이미지 채널 수 (RGB: 3)
    "qkv_bias": True,          # Q, K, V 프로젝션 레이어에 bias 사용 여부
    "use_faster_attention": True,  # FasterMultiHeadAttention 사용 여부
}

# 설정값 오류 사전 방지용 검증
assert config["hidden_size"] % config["num_attention_heads"] == 0  # 헤드 크기가 나눠 떨어져야 함
assert config['intermediate_size'] == 4 * config['hidden_size']    # MLP 확장 비율 확인
assert config['image_size'] % config['patch_size'] == 0            # 이미지가 패치로 딱 나눠져야 함


# ============================================================
# 학습 루프 관리 클래스
# ============================================================
class Trainer:
    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)  # 모델을 GPU/CPU로 이동
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name       # 실험 이름 (결과 저장 폴더명으로 사용)
        self.device = device

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        train_losses, test_losses, accuracies = [], [], []

        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

            # N 에폭마다 중간 체크포인트 저장 (마지막 에폭 제외)
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                save_checkpoint(self.exp_name, self.model, i+1)

        # 학습 완료 후 config, metrics, 모델 가중치 저장
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """배치 단위로 순전파 → 손실 계산 → 역전파 → 가중치 업데이트"""
        self.model.train()  # 드롭아웃 등 학습 모드 활성화
        total_loss = 0

        for batch in trainloader:
            batch = [t.to(self.device) for t in batch]  # 배치를 GPU/CPU로 이동
            images, labels = batch

            self.optimizer.zero_grad()           # 이전 배치의 기울기 초기화

            loss = self.loss_fn(self.model(images)[0], labels)  # [0]: logits만 사용

            loss.backward()                      # 역전파로 기울기 계산
            self.optimizer.step()                # 가중치 업데이트

            total_loss += loss.item() * len(images)  # 배치 크기로 가중합산

        return total_loss / len(trainloader.dataset)  # 샘플 평균 손실 반환

    @torch.no_grad()  # 평가 시에는 기울기 계산 불필요 → 메모리/속도 절약
    def evaluate(self, testloader):
        """테스트셋 전체에 대해 손실과 정확도 계산"""
        self.model.eval()  # 드롭아웃 등 추론 모드로 전환
        total_loss = 0
        correct = 0

        with torch.no_grad():
            for batch in testloader:
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                logits, _ = self.model(images)  # 어텐션 맵은 필요 없으므로 _ 로 버림

                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # 가장 높은 logit 값의 인덱스 = 예측 클래스
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()

        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)   # 실험 이름 (필수)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str)                     # 미지정 시 자동 감지
    parser.add_argument("--save-model-every", type=int, default=0)# 0이면 중간 저장 안 함

    args = parser.parse_args()
    # device 미지정 시 GPU 있으면 cuda, 없으면 cpu 자동 선택
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main():
    args = parse_args()

    # CIFAR-10 데이터 로드
    trainloader, testloader, _ = prepare_data(batch_size=args.batch_size)

    # 모델 생성
    model = ViTForClassfication(config)

    # AdamW: Adam + 가중치 감쇠(L2 정규화). 트랜스포머 학습의 표준 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # CrossEntropyLoss: 다중 분류의 표준 손실 함수 (softmax + NLL loss 합친 것)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device=args.device)
    trainer.train(trainloader, testloader, args.epochs, save_model_every_n_epochs=args.save_model_every)


if __name__ == "__main__":
    main()
