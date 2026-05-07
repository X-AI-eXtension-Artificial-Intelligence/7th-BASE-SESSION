import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import model
import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 전역 변수 설정 ---
PAD_idx = 0
SOS_token = 0
EOS_token = 1
hidden_size = 256
batch_size = 32

# --- 메인 학습 루프 ---
def train(train_dataloader, model, n_epochs, learning_rate=0.0003):
    # 최적화 알고리즘 (Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 손실 함수 (NLLLoss): 패딩 토큰(PAD_idx)에 대해서는 손실을 무시하도록 설정
    criterion = nn.NLLLoss(ignore_index=PAD_idx)

    for epoch in range(1, n_epochs + 1):
        loss = 0
        # 배치(Batch) 단위로 학습 진행
        for iter, batch in enumerate(train_dataloader):
            input_tensor  = batch[0]
            input_mask    = batch[1]
            target_tensor = batch[2]
            
            # 단일 배치 단위의 학습 스텝 수행
            loss += train_step(input_tensor, input_mask, target_tensor,
                               model, optimizer, criterion)
        print('Epoch {} Loss {}'.format(epoch, loss / iter))


# --- 단일 배치 학습 스텝 ---
def train_step(input_tensor, input_mask, target_tensor, model,
               optimizer, criterion):
    # 기울기(gradient) 초기화
    optimizer.zero_grad()
    
    # 모델 순전파 (Forward Pass)
    decoder_outputs, decoder_hidden = model(input_tensor, input_mask, target_tensor)

    # NLLLoss 계산을 위해 차원 축소 (Batch * SeqLen 형태로 펼침)
    loss = criterion(
        decoder_outputs.view(-1, decoder_outputs.size(-1)), 
        target_tensor.view(-1) 
    )

    # 역전파 (Backward Pass) 및 가중치 업데이트
    loss.backward()
    optimizer.step()
    return loss.item()

# --- 토큰 인덱스를 실제 단어 문자열로 변환하는 헬퍼 함수 ---
def ids2words(lang, ids):
    return [lang.index2word[idx] for idx in ids]

# --- 추론/디코딩 (Greedy Search) ---
def greedy_decode(model, dataloader, input_lang, output_lang):
    with torch.no_grad(): # 역전파 비활성화 (메모리 절약 및 속도 향상)
        # 데이터로더에서 첫 번째 배치를 가져옴
        batch = next(iter(dataloader))
        input_tensor  = batch[0]
        input_mask    = batch[1]
        target_tensor = batch[2]

        # 타겟을 주지 않고(None) 모델 실행 -> 모델이 스스로 다음 단어 예측(Greedy)
        decoder_outputs, decoder_hidden = model(input_tensor, input_mask)
        # 가장 확률이 높은 단어의 인덱스(topi) 추출
        topv, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        # 배치 내의 각 문장들에 대해 출력 비교
        for idx in range(input_tensor.size(0)):
            input_sent = ids2words(input_lang, input_tensor[idx].cpu().numpy())
            output_sent = ids2words(output_lang, decoded_ids[idx].cpu().numpy())
            target_sent = ids2words(output_lang, target_tensor[idx].cpu().numpy())
            
            print('Input:  {}'.format(input_sent))
            print('Target: {}'.format(target_sent))
            print('Output: {}'.format(output_sent))

# --- 실행부 (Entry Point) ---
if __name__ == '__main__':
    # 1. 데이터 로드 및 어휘 사전/데이터로더 생성
    input_lang, output_lang, train_dataloader = load_data.get_dataloader(batch_size)
    # 2. 모델 인스턴스화 및 디바이스(GPU/CPU) 할당
    model = model.EncoderDecoder(hidden_size, input_lang.n_words, output_lang.n_words).to(device)
    # 3. 모델 학습 진행 (20 Epochs)
    train(train_dataloader, model, n_epochs=20)
    # 4. 학습된 모델 결과 확인
    greedy_decode(model, train_dataloader, input_lang, output_lang)