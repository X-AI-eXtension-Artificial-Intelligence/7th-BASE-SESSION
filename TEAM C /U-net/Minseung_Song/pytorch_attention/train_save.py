"""train.py와 동일하되, 학습 완료 후 모델과 언어 데이터를 저장하는 버전.
원본 train.py는 학습만 하고 끝나서 모델이 메모리에서 사라진다.
이 파일은 학습 후 모델 가중치와 단어 사전을 파일로 저장해서,
나중에 inference.py에서 불러와 직접 번역 테스트를 할 수 있게 한다.
"""
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pickle
import model
import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_idx = 0
SOS_token = 0
EOS_token = 1
hidden_size = 256
batch_size = 32

def train(train_dataloader, model, n_epochs, learning_rate=0.0003):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_idx)

    for epoch in range(1, n_epochs + 1):
        loss = 0
        for iter, batch in enumerate(train_dataloader):
            input_tensor  = batch[0]
            input_mask    = batch[1]
            target_tensor = batch[2]
            loss += train_step(input_tensor, input_mask, target_tensor,
                               model, optimizer, criterion)
        print('Epoch {} Loss {}'.format(epoch, loss / iter))


def train_step(input_tensor, input_mask, target_tensor, model,
               optimizer, criterion):
    optimizer.zero_grad()
    decoder_outputs, decoder_hidden = model(input_tensor, input_mask, target_tensor)

    loss = criterion(
        decoder_outputs.view(-1, decoder_outputs.size(-1)),
        target_tensor.view(-1)
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
    enc_dec = model.EncoderDecoder(hidden_size, input_lang.n_words, output_lang.n_words).to(device)
    train(train_dataloader, enc_dec, n_epochs=20)
    greedy_decode(enc_dec, train_dataloader, input_lang, output_lang)

    # ===== 모델 및 언어 데이터 저장 =====
    # 모델 가중치 저장 (.pt 파일)
    torch.save({
        'model_state_dict': enc_dec.state_dict(),
        'hidden_size': hidden_size,
        'input_vocab_size': input_lang.n_words,
        'output_vocab_size': output_lang.n_words,
    }, 'attention_model.pt')

    # 단어 사전 저장 (pickle)
    # 모델만 저장하면 숫자<->단어 매핑을 모르니까, Lang 객체도 같이 저장해야 한다.
    with open('lang_data.pkl', 'wb') as f:
        pickle.dump({
            'input_lang': input_lang,
            'output_lang': output_lang,
        }, f)

    print('\n모델 저장 완료: attention_model.pt')
    print('언어 데이터 저장 완료: lang_data.pkl')
