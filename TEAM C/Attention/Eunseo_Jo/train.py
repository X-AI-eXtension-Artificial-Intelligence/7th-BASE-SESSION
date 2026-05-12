import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import model
import load_data

#cuda가 잇으면 쓰고(gpu) else cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_idx = 0
SOS_token = 0
EOS_token = 1
hidden_size = 256
batch_size = 32

def train(train_dataloader, model, n_epochs, learning_rate=0.0003):
    # Adam 최적화 알고리즘 사용
    # Momentom 이 갖고있는 가속도 (가는 방향으로 더 힘줘서 가서 안장점을 탈출)
    # RMSProp 은 진동을 줄여주고 효율적으로 학습하게 해주는 브레이크
    # (학습이 많이되면 조금되도록 줄이고, 학습이 너무 늘이면 더 많이!)
    # 이 두가지의 장점을 합쳐둔게 Adam 이고, 따라서 안장점과 진동을 줄여줌
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_idx)

    for epoch in range(1, n_epochs + 1):
        loss = 0
        for iter, batch in enumerate(train_dataloader):
            # Batch tensors: [B, SeqLen]
            input_tensor  = batch[0]
            input_mask    = batch[1]
            target_tensor = batch[2]
            loss += train_step(input_tensor, input_mask, target_tensor,
                               model, optimizer, criterion)
        print('Epoch {} Loss {}'.format(epoch, loss / iter))

# 역전파 학습단계
def train_step(input_tensor, input_mask, target_tensor, model,
               optimizer, criterion):
    optimizer.zero_grad()
    decoder_outputs, decoder_hidden = model(input_tensor, input_mask, target_tensor)

    # Collapse [B, Seq] dimensions for NLL Loss
    loss = criterion(
        decoder_outputs.view(-1, decoder_outputs.size(-1)), # [B, Seq, OutVoc] -> [B*Seq, OutVoc]
        target_tensor.view(-1) # [B, Seq] -> [B*Seq]
    )

    loss.backward()
    optimizer.step()
    return loss.item()

def ids2words(lang, ids):
    return [lang.index2word[idx] for idx in ids]

# Greedy 기법: 각 시점에서 가장 확률이 높은(top 1) 단어 선택
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


if __name__ == '__main__':
    input_lang, output_lang, train_dataloader = load_data.get_dataloader(batch_size)
    model = model.EncoderDecoder(hidden_size, input_lang.n_words, output_lang.n_words).to(device)
    train(train_dataloader, model, n_epochs=20)
    greedy_decode(model, train_dataloader, input_lang, output_lang)


