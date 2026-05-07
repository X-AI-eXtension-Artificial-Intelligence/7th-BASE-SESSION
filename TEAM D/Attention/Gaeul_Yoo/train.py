import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import model
import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_idx = 0
SOS_token = 0
EOS_token = 1
hidden_size = 256
batch_size = 32

def train(train_dataloader, model, n_epochs, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 성능 최적화 포인트: 에포크마다 학습률을 0.95배씩 감소시킴
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        for data in train_dataloader:
            input_tensor, input_mask, target_tensor, target_mask = data
            
            optimizer.zero_grad()
            decoder_outputs, _ = model(input_tensor, input_mask, target_tensor)
            
            # target_mask를 활용해 패딩된 부분은 손실 계산에서 제외
            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step() # 에포크 종료 후 학습률 갱신
        print(f'Epoch {epoch} Loss {total_loss / len(train_dataloader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')

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