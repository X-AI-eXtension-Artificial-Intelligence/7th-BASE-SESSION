"""
train.py
- Transformer 모델을 만들고 학습/평가를 수행하는 메인 파일입니다.
"""

import math
import time

import torch
from torch import nn, optim

from conf import *
from data import *
from graph import draw
from models.model.transformer import Transformer
from util.bleu import get_bleu, idx_to_word
from util.epoch_timer import epoch_time


# 전체 Transformer 모델을 생성합니다.
model = Transformer(
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    trg_sos_idx=trg_sos_idx,
    enc_voc_size=enc_voc_size,
    dec_voc_size=dec_voc_size,
    d_model=d_model,
    n_head=n_head,
    max_len=max_len,
    ffn_hidden=ffn_hidden,
    n_layers=n_layers,
    drop_prob=drop_prob,
    device=device
).to(device)

# Adam optimizer입니다.
# 논문식 warmup scheduler는 없고, 고정 learning rate를 사용합니다.
optimizer = optim.Adam(
    params=model.parameters(),
    lr=init_lr,
    weight_decay=weight_decay,
    eps=adam_eps
)

# target vocabulary에 대한 multi-class classification loss입니다.
# 일반적으로 target padding index를 ignore_index로 둡니다.
# 원 코드에서는 src_pad_idx를 사용하지만, source/target pad index가 같으면 문제없이 동작할 수 있습니다.
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def count_parameters(model):
    """학습 가능한 파라미터 수를 계산합니다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    """모델의 weight tensor를 Xavier uniform으로 초기화합니다."""
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, criterion, clip):
    """한 epoch 동안 학습을 수행합니다."""
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        # batch.src, batch.trg는 torchtext Field 이름에 의해 생성됩니다.
        src = batch.src
        trg = batch.trg

        # 이전 batch의 gradient를 초기화합니다.
        optimizer.zero_grad()

        # teacher forcing 방식입니다.
        # decoder 입력: <sos>부터 마지막 직전 token까지
        output = model(src, trg[:, :-1])

        # 정답: 첫 token 다음부터 <eos>까지
        # output shape: [batch, trg_len-1, dec_voc_size]
        # trg shape: [batch, trg_len-1]
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        # loss 계산 후 역전파합니다.
        loss = criterion(output_reshape, trg)
        loss.backward()

        # gradient clipping으로 gradient exploding을 방지합니다.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # parameter update입니다.
        optimizer.step()

        epoch_loss += loss.item()

        # 일정 간격마다 현재 batch loss를 출력합니다.
        if i % clip == 0:
            print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    """validation/test loss와 BLEU를 계산합니다."""
    model.eval()
    epoch_loss = 0
    batch_bleu = []

    # 평가 단계에서는 gradient 계산이 필요 없습니다.
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # 학습 때와 동일하게 target을 한 칸 밀어서 넣습니다.
            output = model(src, trg[:, :-1])

            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_reshape = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg_reshape)
            epoch_loss += loss.item()

            # BLEU 계산을 위해 모델 예측 index와 실제 index를 단어 문자열로 변환합니다.
            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except Exception:
                    # 마지막 batch는 batch_size보다 작을 수 있으므로 예외가 날 수 있습니다.
                    pass

            batch_bleu.append(sum(total_bleu) / len(total_bleu))

    return epoch_loss / len(iterator), sum(batch_bleu) / len(batch_bleu)


def run(total_epoch, best_loss):
    """전체 epoch 학습을 실행하고, validation loss가 좋아지면 모델을 저장합니다."""
    train_losses, test_losses, bleus = [], [], []

    for step in range(total_epoch):
        start_time = time.time()

        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # validation loss가 가장 낮은 모델을 저장합니다.
        if step > int(total_epoch / 2) and valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), model_path.format(valid_loss))

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')

    # 학습 종료 후 loss 그래프를 저장합니다.
    draw(mode='loss', train=train_losses, val=test_losses, path='loss.png')


if __name__ == '__main__':
    # 파라미터 수를 출력합니다.
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # 가중치 초기화입니다.
    model.apply(initialize_weights)

    # 매우 큰 초기 best_loss로 시작합니다.
    best_loss = float('inf')

    # 전체 학습 실행입니다.
    run(total_epoch=epochs, best_loss=best_loss)
