import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import model
import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_idx = 0       # SOS_token과 같은 값. 분리하는 게 안전한 디자인
SOS_token = 0
EOS_token = 1
hidden_size = 256     # 논문은 1000. toy니까 작게
batch_size = 32


def train(train_dataloader, model, n_epochs, learning_rate=0.0003):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)   # 논문은 Adadelta
    criterion = nn.NLLLoss(ignore_index=PAD_idx)                   # padding 위치 loss 무시

    for epoch in range(1, n_epochs + 1):
        loss = 0
        for iter, batch in enumerate(train_dataloader):
            input_tensor  = batch[0]
            input_mask    = batch[1]
            target_tensor = batch[2]
            # target_mask는 안 씀 (NLLLoss의 ignore_index로 padding 처리하니까)
            loss += train_step(input_tensor, input_mask, target_tensor,
                               model, optimizer, criterion)
        # iter는 마지막 batch index. len(dataloader)로 나누는 게 더 정확
        print('Epoch {} Loss {}'.format(epoch, loss / iter))

    # NOTE: gradient clipping 없음. 논문은 norm 1로 clip. RNN에서 보통 필요
    # NOTE: validation loop도 없음. overfitting 모니터링 불가


def train_step(input_tensor, input_mask, target_tensor, model,
               optimizer, criterion):
    optimizer.zero_grad()

    # target_tensor를 넘기면 decoder 내부에서 teacher forcing 작동
    decoder_outputs, decoder_hidden = model(input_tensor, input_mask, target_tensor)

    # NLLLoss 입력 형식 맞추기:
    #   decoder_outputs: [B, Seq, OutVocab]  ->  [B*Seq, OutVocab]
    #   target_tensor:   [B, Seq]            ->  [B*Seq]
    # 모든 (batch, position) 쌍에 대한 token-level loss를 한 번에 계산
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
    """
    Inference. target_tensor 안 넘김 -> teacher forcing 안 걸림 -> 자기 예측을 다음 입력으로
    """
    with torch.no_grad():
        batch = next(iter(dataloader))     # 첫 batch만 봄
        input_tensor  = batch[0]
        input_mask    = batch[1]
        target_tensor = batch[2]

        # target 안 넘김 -> AttnDecoder.forward의 else 분기 (greedy) 작동
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

    # 한계:
    # - greedy만. 논문은 beam search
    # - EOS 만나도 안 멈춤. max_len까지 계속 생성 (EOS 이후는 noise)
    # - training data로 evaluate. validation set 분리 안 됨


if __name__ == '__main__':
    input_lang, output_lang, train_dataloader = load_data.get_dataloader(batch_size)
    model = model.EncoderDecoder(hidden_size, input_lang.n_words, output_lang.n_words).to(device)
    train(train_dataloader, model, n_epochs=20)
    greedy_decode(model, train_dataloader, input_lang, output_lang)
