"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""

'''
실행방법: 아래 명령어를 터미널에 입력
pip install torch sacrebleu
cd transformer_project
python train.py
'''
import math
import time
import os

import torch
from torch import nn, optim

from conf import *
from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model = Transformer(
    src_pad_idx  = src_pad_idx,
    trg_pad_idx  = trg_pad_idx,
    trg_sos_idx  = trg_sos_idx,
    d_model      = d_model,
    enc_voc_size = enc_voc_size,
    dec_voc_size = dec_voc_size,
    max_len      = max_len,
    ffn_hidden   = ffn_hidden,
    n_head       = n_heads,
    n_layers     = n_layers,
    drop_prob    = drop_prob,
    device       = device,
).to(device)

print(f'파라미터 수: {count_parameters(model):,}')
model.apply(initialize_weights)

optimizer = optim.Adam(
    params=model.parameters(),
    lr=init_lr,
    weight_decay=weight_decay,
    eps=adam_eps,
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    factor=factor,
    patience=patience,
)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        optimizer.zero_grad()
        output         = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg_y          = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        print(f'\rstep: {round((i / len(iterator)) * 100, 1)}%  loss: {loss.item():.4f}', end='')

    print()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss  = 0
    hypotheses  = []
    references  = []

    with torch.no_grad():
        for src, trg in iterator:
            output         = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_y          = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg_y)
            epoch_loss += loss.item()

            preds = output.argmax(dim=-1)
            for j in range(src.size(0)):
                references.append(idx_to_word(trg[j, 1:], loader.target))
                hypotheses.append(idx_to_word(preds[j],   loader.target))

    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    except Exception:
        bleu = 0.0

    return epoch_loss / len(iterator), bleu


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    os.makedirs('result', exist_ok=True)
    os.makedirs('saved',  exist_ok=True)

    print('\n' + '=' * 68)
    print(f'  Transformer EN→DE | device={device} | epochs={total_epoch}')
    print('=' * 68)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>9} | {'BLEU':>6} | {'Time':>7}")
    print('-' * 68)

    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time   = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)

        mins, secs = epoch_time(start_time, end_time)
        marker = ' ★' if valid_loss < best_loss else ''

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'saved/model-{valid_loss:.4f}.pt')

        print(f'{step+1:>6} | {train_loss:>10.4f} | {valid_loss:>9.4f} | {bleu:>6.2f} | {mins}m{secs:02d}s{marker}')

        with open('result/train_loss.txt', 'w') as f:
            f.write(str(train_losses))
        with open('result/test_loss.txt', 'w') as f:
            f.write(str(test_losses))
        with open('result/bleu.txt', 'w') as f:
            f.write(str(bleus))

    print('=' * 68)
    print(f'\n✅ 학습 완료!  Best Val Loss: {best_loss:.4f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
