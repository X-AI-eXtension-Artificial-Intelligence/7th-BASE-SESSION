"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time

# 학습 가능한 파라미터 수 계싼
# p.requires_grad가 True인 파라미터만 사용함
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 모델 가중치 초기화 함수
# 모델 내부 layer의 weight를 초기화함
# weight가 2차원 이상인지 확인함
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
        # Kaiming_uniform 방식으로 weight 초기화


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

# 모델 파라미터 수 출력 밒 초기화 
# 첫 줄은 학습 가능한 파라미터 수 출력
print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
# initaialize_weight () 함수 적용
# transformer 안의 여러 layer들의 weight
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)
# adam optimizer를 사용

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)
# learning rate scheduler 생성(낮춤)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
# loss 함수 생성, 보통 위의 함수를 사용해서 계산함

def train(model, iterator, optimizer, criterion, clip):
    # Dropout 등이 학습 모드로 동작함
    model.train()
    # 한 epoch 동안의 loss를 누적할 변수
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        # source 문장 데이터
        src = batch.src
        # target 문장 데이터
        trg = batch.trg

        optimizer.zero_grad()
        # Transformer 모델에 source와 target 입력을 넣어 예측 수행
        # trg[:, :-1]은 target 문장에서 마지막 토큰을 제외한 입력
        output = model(src, trg[:, :-1])
        # CrossEntropyLoss 계산을 위해 output 형태 변경
        # 변경 shape: [batch_size * (trg_len - 1), vocab_size]
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        # 정답 target에서 첫 토큰 <sos>를 제외
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    # Dropout 등이 비활성화됨
    model.eval()
    # validation/test loss를 누적할 변수
    epoch_loss = 0
    # batch별 BLEU score를 저장할 리스트
    batch_bleu = []
    # 평가 단계에서는 gradient 계산이 필요 없으므로 비활성화
    with torch.no_grad():
        # iterator에서 batch를 하나씩 꺼내면서 반복
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)
            # validation/test loss 계산
            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            # batch 안의 각 문장에 대해 BLEU score 계산
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    # output에서 각 위치마다 가장 확률이 높은 단어 index 선택
                    output_words = output[j].max(dim=1)[1]
                    # 예측 index들을 실제 단어 문장으로 변환
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    # 마지막 batch 크기가 batch_size보다 작거나
                    # BLEU 계산 중 오류가 나면 해당 문장은 건너뜀
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        # warmup 이후부터 learning rate scheduler 적용
        # validation loss가 좋아지지 않으면 learning rate를 줄임
        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
         # epoch 수행 시간을 분, 초 단위로 변환
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')

# 이 파일을 직접 실행했을 때만 아래 코드 실행
if __name__ == '__main__':
    # 전체 epoch 수만큼 학습 시작
    # best_loss는 무한대로 시작해서 첫 validation loss가 저장되도록 함
    run(total_epoch=epoch, best_loss=inf)
