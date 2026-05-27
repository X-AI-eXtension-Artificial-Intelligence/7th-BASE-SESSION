import torch
from torch import nn, optim

from utils import save_experiment, save_checkpoint
from data import prepare_data
from vit import ViTForClassfication

# ViT 모델의 하이퍼파라미터 설정
config = {
    "patch_size": 4,  # 입력 이미지(32x32)를 8x8 개의 패치로 나눔 (32/4=8)
    "hidden_size": 48, # 임베딩 및 은닉층의 차원 크기
    "num_hidden_layers": 4, # 트랜스포머 인코더 블록의 개수
    "num_attention_heads": 4, # 멀티 헤드 어텐션의 헤드 개수
    "intermediate_size": 4 * 48, # MLP 내부의 은닉층 크기 (보통 hidden_size의 4배)
    "hidden_dropout_prob": 0.0, # 드롭아웃 비율
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02, # 가중치 초기화 범위
    "image_size": 32, # 입력 이미지 크기
    "num_classes": 10, # CIFAR-10 클래스 개수
    "num_channels": 3, # RGB 채널 수
    "qkv_bias": True, # Q, K, V 계산 시 편향(bias) 사용 여부
    "use_faster_attention": True, # 최적화된 어텐션 연산 사용 여부
}

# 설정값이 올바른지 검증 (강제 조건은 아니지만 오류 방지용)
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0


class Trainer:
    """
    모델 학습과 평가를 담당하는 Trainer 클래스
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        지정된 에포크 수만큼 모델을 학습
        """
        # 손실과 정확도를 기록할 리스트 초기화
        train_losses, test_losses, accuracies = [], [], []
        
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader) # 1 에포크 학습
            accuracy, test_loss = self.evaluate(testloader) # 평가 진행
            
            # 기록 저장
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # 설정된 주기마다 모델 가중치(체크포인트) 중간 저장
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                save_checkpoint(self.exp_name, self.model, i+1)
        
        # 전체 학습이 끝나면 최종 실험 결과 저장
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """
        1 에포크 동안의 학습 과정 (순전파 및 역전파)
        """
        self.model.train() # 모델을 학습 모드로 설정
        total_loss = 0
        for batch in trainloader:
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            
            self.optimizer.zero_grad() # 이전 배치의 기울기 초기화
            loss = self.loss_fn(self.model(images)[0], labels) # 순전파 및 손실 계산
            loss.backward() # 역전파로 기울기 계산
            self.optimizer.step() # 가중치 업데이트
            
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset) # 평균 에포크 손실 반환

    @torch.no_grad()
    def evaluate(self, testloader):
        """
        테스트 데이터셋을 이용한 모델 성능 평가
        """
        self.model.eval() # 모델을 평가 모드로 설정 (드롭아웃 등 비활성화)
        total_loss = 0
        correct = 0
        with torch.no_grad(): # 평가 시에는 기울기 계산을 하지 않음 (메모리 절약)
            for batch in testloader:
                batch = [t.to(self.device) for t in batch]
                images, labels = batch
                
                logits, _ = self.model(images) # 예측값 도출
                loss = self.loss_fn(logits, labels) # 손실 계산
                total_loss += loss.item() * len(images)

                # 가장 높은 확률을 가진 클래스를 예측 결과로 선택
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item() # 정답 개수 누적
                
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def parse_args():
    # 커맨드라인에서 실행 인자를 받을 수 있도록 파서 설정
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-model-every", type=int, default=0)

    args = parser.parse_args()
    # GPU가 있으면 cuda를, 없으면 cpu를 사용하도록 기본값 설정
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main():
    args = parse_args()
    
    # 학습 파라미터 가져오기
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    save_model_every_n_epochs = args.save_model_every
    
    # 데이터셋 준비
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)
    
    # 모델, 최적화 알고리즘, 손실 함수 초기화
    model = ViTForClassfication(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    
    # 트레이너 객체 생성 및 학습 시작
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device=device)
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)

if __name__ == "__main__":
    main()