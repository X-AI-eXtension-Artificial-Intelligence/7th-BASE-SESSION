# PyTorch 관련 라이브러리 불러오기
import torch
from torch import nn, optim

# 실험 저장, 데이터 준비, ViT 모델 클래스 불러오기
from utils import save_experiment, save_checkpoint
from data import prepare_data
from vit import ViTForClassfication


# ViT 모델 학습에 사용할 설정값
config = {
    "patch_size": 4,  # 입력 이미지 32x32를 4x4 패치로 나눔 -> 8x8 = 64개 패치
    "hidden_size": 48,  # 각 패치를 표현할 임베딩 벡터 차원
    "num_hidden_layers": 4,  # Transformer Encoder Block 개수
    "num_attention_heads": 4,  # Multi-head Attention에서 사용할 head 개수
    "intermediate_size": 4 * 48, # MLP 내부 은닉층 크기, 일반적으로 hidden_size의 4배 사용
    "hidden_dropout_prob": 0.0,  # hidden layer 출력에 적용할 dropout 비율
    "attention_probs_dropout_prob": 0.0,  # attention probability에 적용할 dropout 비율
    "initializer_range": 0.02,  # 가중치 초기화 시 사용할 표준편차
    "image_size": 32,  # 입력 이미지 크기
    "num_classes": 10, # CIFAR-10은 총 10개 클래스
    "num_channels": 3,  # RGB 이미지이므로 채널 수는 3
    "qkv_bias": True,  # Query, Key, Value 선형 변환에 bias 사용 여부
    "use_faster_attention": True,  # 최적화된 Multi-head Attention 사용 여부
}

# 잘못된 설정으로 모델이 생성되는 것을 방지하기 위한 확인 코드
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0


class Trainer:
    """
    모델 학습과 평가 과정을 관리하는 간단한 Trainer 클래스
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        # 모델을 사용할 장치로 이동
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        지정한 epoch 수만큼 모델을 학습하는 함수
        """
        # epoch별 학습 loss, 테스트 loss, 정확도를 저장할 리스트
        train_losses, test_losses, accuracies = [], [], []

        # 전체 epoch 반복
        for i in range(epochs):
            # 한 epoch 학습 진행
            train_loss = self.train_epoch(trainloader)

            # 테스트 데이터로 모델 평가
            accuracy, test_loss = self.evaluate(testloader)

            # 결과 저장
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            # 현재 epoch 결과 출력
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

            # 설정한 주기마다 중간 checkpoint 저장
            # 마지막 epoch에서는 아래 save_experiment에서 최종 모델을 따로 저장하므로 제외
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                save_checkpoint(self.exp_name, self.model, i+1)

        # 학습이 끝난 뒤 설정값, 성능 기록, 최종 모델 저장
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """
        학습 데이터 전체를 한 번 순회하며 모델을 학습하는 함수
        """
        # 모델을 학습 모드로 전환
        self.model.train()
        total_loss = 0

        # DataLoader에서 배치 단위로 이미지와 라벨을 가져옴
        for batch in trainloader:
            # 배치 데이터를 GPU 또는 CPU로 이동
            batch = [t.to(self.device) for t in batch]
            images, labels = batch

            # 이전 배치에서 계산된 gradient 초기화
            self.optimizer.zero_grad()

            # 모델 예측값과 실제 라벨을 비교해 loss 계산
            # self.model(images)[0]은 logits를 의미
            loss = self.loss_fn(self.model(images)[0], labels)

            # loss를 기준으로 역전파 수행
            loss.backward()

            # optimizer가 모델 파라미터 업데이트
            self.optimizer.step()

            # 배치 loss를 샘플 수만큼 곱해 누적
            total_loss += loss.item() * len(images)

        # 데이터셋 전체 기준 평균 loss 반환
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        """
        테스트 데이터로 모델 성능을 평가하는 함수
        """
        # 모델을 평가 모드로 전환
        self.model.eval()
        total_loss = 0
        correct = 0

        # 평가 과정에서는 gradient 계산이 필요 없으므로 비활성화
        with torch.no_grad():
            for batch in testloader:
                # 배치 데이터를 GPU 또는 CPU로 이동
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                # 모델 예측 수행
                logits, _ = self.model(images)

                # 테스트 loss 계산
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # 가장 높은 logit 값을 가진 클래스를 예측 클래스로 선택
                predictions = torch.argmax(logits, dim=1)

                # 정답 라벨과 예측 라벨이 같은 개수 누적
                correct += torch.sum(predictions == labels).item()

        # 전체 테스트 데이터 기준 정확도와 평균 loss 계산
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def parse_args():
    """
    터미널에서 입력받을 학습 옵션을 정의하고 파싱하는 함수
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)  # 실험 이름
    parser.add_argument("--batch-size", type=int, default=256)  # 배치 크기
    parser.add_argument("--epochs", type=int, default=100)  # 학습 epoch 수
    parser.add_argument("--lr", type=float, default=1e-2)  # learning rate
    parser.add_argument("--device", type=str)  # 사용할 장치, 지정하지 않으면 자동 선택
    parser.add_argument("--save-model-every", type=int, default=0)  # checkpoint 저장 주기

    args = parser.parse_args()

    # device를 직접 지정하지 않은 경우, CUDA 사용 가능 여부에 따라 자동 설정
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main():
    """
    학습 실행의 전체 흐름을 담당하는 메인 함수
    """
    # 터미널 인자 읽기
    args = parse_args()

    # 학습 파라미터 설정
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    save_model_every_n_epochs = args.save_model_every

    # CIFAR-10 데이터셋 로드
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)

    # 모델, optimizer, loss function, trainer 생성
    model = ViTForClassfication(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device=device)

    # 학습 시작
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)


# 이 파일을 직접 실행했을 때만 main 함수 실행
if __name__ == "__main__":
    main()
