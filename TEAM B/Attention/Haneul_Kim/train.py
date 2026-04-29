import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import model
import load_data

# GPU 사용 안되면 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# padding 토큰 인덱스 (loss 계산 시 무시)
PAD_idx = 0
# 문장 시작 / 끝 토큰
SOS_token = 0
EOS_token = 1
# 모델 하이퍼파라미터
hidden_size = 256
batch_size = 32

# 전체 학습
def train(train_dataloader, model, n_epochs, learning_rate=0.0003):
    # Adam optimizer 사용
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # NLLLoss 사용, padding 위치는 loss 계산에서 제외
    criterion = nn.NLLLoss(ignore_index=PAD_idx)

    for epoch in range(1, n_epochs + 1):
        loss = 0  # epoch 전체 loss 누적
        for iter, batch in enumerate(train_dataloader):
            # Batch tensors: [B, SeqLen]
            input_tensor  = batch[0]
            input_mask    = batch[1]
            target_tensor = batch[2]
            
            # 한 step 학습 수행
            loss += train_step(input_tensor, input_mask, target_tensor,
                               model, optimizer, criterion)
        # epoch 평균 loss 출력
        print('Epoch {} Loss {}'.format(epoch, loss / iter))

# 한 step 학습
def train_step(input_tensor, input_mask, target_tensor, model,
               optimizer, criterion):
    
    # 이전 gradient 초기화
    optimizer.zero_grad()
    
    # 모델 실행 (Encoder → Decoder)
    decoder_outputs, decoder_hidden = model(input_tensor, input_mask, target_tensor)

    # Collapse [B, Seq] dimensions for NLL Loss
    loss = criterion(
        decoder_outputs.view(-1, decoder_outputs.size(-1)), # [B, Seq, OutVoc] -> [B*Seq, OutVoc]
        target_tensor.view(-1) # [B, Seq] -> [B*Seq]
    )
    # Backpropagation
    loss.backward() # gradient 계산
    optimizer.step() # 파라미터 업뎃
    return loss.item()

# 인덱스를 단어로 변환
def ids2words(lang, ids):
    return [lang.index2word[idx] for idx in ids]

# Greedy decoding
def greedy_decode(model, dataloader, input_lang, output_lang):
    with torch.no_grad():
        batch = next(iter(dataloader))
        input_tensor  = batch[0]
        input_mask    = batch[1]
        target_tensor = batch[2]

        decoder_outputs, decoder_hidden = model(input_tensor, input_mask)
        topv, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        for idx in range(input_tensor.size(0)):
            input_sent = ids2words(input_lang, input_tensor[idx].cpu().numpy())
            output_sent = ids2words(output_lang, decoded_ids[idx].cpu().numpy())
            target_sent = ids2words(output_lang, target_tensor[idx].cpu().numpy())
            print('Input:  {}'.format(input_sent))
            print('Target: {}'.format(target_sent))
            print('Output: {}'.format(output_sent))

# 메인 실행
if __name__ == '__main__':
    # 데이터 로딩 (DataLoader 생성)
    input_lang, output_lang, train_dataloader = load_data.get_dataloader(batch_size)
    
    # Encoder-Decoder 모델 생성
    model = model.EncoderDecoder(hidden_size, input_lang.n_words, output_lang.n_words).to(device)
    
    # 학습
    train(train_dataloader, model, n_epochs=20)
    
    # 학습된 모델로 번역 결과 확인
    greedy_decode(model, train_dataloader, input_lang, output_lang)


