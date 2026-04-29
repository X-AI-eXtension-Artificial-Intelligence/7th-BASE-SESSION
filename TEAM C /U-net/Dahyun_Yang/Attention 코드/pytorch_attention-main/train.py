import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import model
import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_idx = 0 # 패딩 인덱스: 손실 계산 시 무시할 대상
SOS_token = 0 
EOS_token = 1 
hidden_size = 256 # 모델의 은닉층 크기
batch_size = 32 # 한 번에 학습할 문장 개수

def train(train_dataloader, model, n_epochs, learning_rate=0.0003):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 최적화 알고리즘
    criterion = nn.NLLLoss(ignore_index=PAD_idx) # 손실 함수: 패딩은 무시하고 예측값과 타겟 비교
    # .
    # optimizer랑 criterion이 루프 안이 아니라 밖에서 정의한 이유는 딥러닝 내에서 매우 중요한 설계 원칙을 따르기 때문
    # optimizer는 단순히 계산만 하는 게 아니라 현재까지의 가중치 업데이터 상태를 내부적으로 기억하고 있음
        # 루프 안에서 매번 새로 생성하면, 중요한 그 전의 학습 정보가 날아가버려서 성능 좋아지지 않음
    
    for epoch in range(1, n_epochs + 1):
        loss = 0
        for iter, batch in enumerate(train_dataloader):
            # Batch tensors: [B, SeqLen]
            input_tensor  = batch[0]
            input_mask    = batch[1]
            target_tensor = batch[2]
            # 한 배치를 학습하고 손실 누적
            loss += train_step(input_tensor, input_mask, target_tensor,
                               model, optimizer, criterion)
        print('Epoch {} Loss {}'.format(epoch, loss / iter))


def train_step(input_tensor, input_mask, target_tensor, model,
               optimizer, criterion):
    optimizer.zero_grad()
    # 앞에서 언급한대로 optimizer을 만드는 것은 무거운 작업이기 떄문에, 모든 파라미터를 받아와서 최적화 객체를 설정하게 되는 일은 낭비!
    # 즉, 루프 밖에서 한번만 설정해두는 것이 훨씬 빠르고 효율적임
    # 모델 순전파
    decoder_outputs, decoder_hidden = model(input_tensor, input_mask, target_tensor)

    # Collapse [B, Seq] dimensions for NLL Loss
    # NLLLoss를 위해 3차원 출력을 2차원으로 변환
    # 모델의 결과와 정답을 비교하여 오차 계산
    loss = criterion(
        decoder_outputs.view(-1, decoder_outputs.size(-1)), # [B, Seq, OutVoc] -> [B*Seq, OutVoc]
        target_tensor.view(-1) # [B, Seq] -> [B*Seq]
    )

    loss.backward() # 역전파: 오차를 통해 가중치 기울기 계산
    optimizer.step() # 가중치 업데이트
    # 절대 루프 안에 넣지 말것!!
    return loss.item()

def ids2words(lang, ids):
    return [lang.index2word[idx] for idx in ids]

def greedy_decode(model, dataloader, input_lang, output_lang):
    # 학습 후, 모델이 잘하는지 눈으로 확인하는 과정이라고 생각하면 됨
    with torch.no_grad(): # 예측 시에는 기울기 계산 안 함
        # 학습하는 과정이 아니기 때문에 굳이 메모리 낭비 ㄴㄴ 생략!
        batch = next(iter(dataloader))
        # 모델 예측
        input_tensor  = batch[0]
        input_mask    = batch[1]
        target_tensor = batch[2]

        decoder_outputs, decoder_hidden = model(input_tensor, input_mask)
        topv, topi = decoder_outputs.topk(1)
        # topv: 가장 높은 확률값(value)
        # topi: 가장 높은 확률을 가진 단어의 인덱스(index)
        decoded_ids = topi.squeeze()
        # 불필요한 차원 제거(위에서 언급한 함수랑 똑같은 함수를 씀!)

        # 인덱스를 단어로 변환하여 출력
        for idx in range(input_tensor.size(0)):
            input_sent = ids2words(input_lang, input_tensor[idx].cpu().numpy())
            output_sent = ids2words(output_lang, decoded_ids[idx].cpu().numpy())
            target_sent = ids2words(output_lang, target_tensor[idx].cpu().numpy())
            print('Input:  {}'.format(input_sent))
            print('Target: {}'.format(target_sent))
            print('Output: {}'.format(output_sent))


if __name__ == '__main__':
    input_lang, output_lang, train_dataloader = load_data.get_dataloader(batch_size)
    model = model.EncoderDecoder(hidden_size, input_lang.n_words, output_lang.n_words).to(device)
    train(train_dataloader, model, n_epochs=20)
    greedy_decode(model, train_dataloader, input_lang, output_lang)


