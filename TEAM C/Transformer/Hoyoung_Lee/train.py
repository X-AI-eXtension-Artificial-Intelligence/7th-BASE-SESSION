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

# 모델의 학습 가능한 파라미터 총 개수를 세는 함수
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 모델 가중치 초기화 함수 (Kaiming Uniform 방식 사용)
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

# --- 1. 모델 설정 및 초기화 ---
model = Transformer(src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, trg_sos_idx=trg_sos_idx,
                    d_model=d_model, enc_voc_size=enc_voc_size, dec_voc_size=dec_voc_size,
                    max_len=max_len, ffn_hidden=ffn_hidden, n_head=n_heads,
                    n_layers=n_layers, drop_prob=drop_prob, device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)

# 옵티마이저 (Adam), 스케줄러(학습률 조절), 손실 함수(CrossEntropy - 패딩 무시) 설정
optimizer = Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=factor, patience=patience)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

# --- 2. 학습(Train) 함수 ---
def train(model, iterator, optimizer, criterion, clip):
    model.train() # 학습 모드 설정 (Dropout 활성화 등)
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad() # 기울기 초기화
        
        # [핵심] Teacher Forcing: 디코더 입력으로는 마지막 단어를 제외(trg[:, :-1])하고 넣음
        output = model(src, trg[:, :-1])
        
        # Loss 계산을 위해 출력 텐서의 형태를 1차원으로 쫙 펼침 (Batch * SeqLen)
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        # 정답(Target)은 첫 번째 토큰(SOS)을 제외(trg[:, 1:])하여, 다음 단어를 예측하도록 정렬
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward() # 역전파
        
        # 기울기 폭발(Gradient Exploding)을 방지하기 위한 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # 가중치 업데이트

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)

# --- 3. 평가(Evaluate) 함수 ---
def evaluate(model, iterator, criterion):
    model.eval() # 평가 모드 설정 (Dropout 비활성화)
    epoch_loss = 0
    batch_bleu = []
    
    with torch.no_grad(): # 역전파 비활성화 (메모리 절약)
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            # 기계 번역 평가 지표인 BLEU 스코어 계산
            total_bleu = []
            for j in range(batch_size):
                try:
                    # 정답 단어들과 모델이 가장 높게 예측한 단어(max)들을 추출하여 비교
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass # 오류 발생 시 무시 (예: 배치 사이즈 불일치 등)

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu

# --- 4. 전체 실행(Run) 루프 ---
def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        
        # 학습 및 평가 진행
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        # 웜업(Warmup) 스텝 이후부터 스케줄러 작동 (학습률 조정)
        if step > warmup:
            scheduler.step(valid_loss)

        # 결과 기록
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 최고 성능 갱신 시 모델의 가중치(State Dict) 저장
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        # 텍스트 파일로 손실 및 BLEU 스코어 로그 저장
        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        # 에폭별 콘솔 출력 (Perplexity(PPL) 포함)
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)