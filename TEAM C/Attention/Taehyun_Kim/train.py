"""
train.py
========
Seq2Seq 모델의 학습, 평가, 추론을 담당하는 메인 스크립트.

실행: python3 train.py
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from collections import Counter
import math
import model
import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────
# 하이퍼파라미터
# ──────────────────────────────────────────
PAD_idx     = 0    # 패딩 토큰 인덱스 (NLLLoss에서 무시)
SOS_token   = 0    # 디코더 시작 토큰
EOS_token   = 1    # 문장 종료 토큰
hidden_size = 256  # GRU hidden 차원
batch_size  = 32   # 배치 크기


# ──────────────────────────────────────────
# 학습 함수
# ──────────────────────────────────────────

def train(train_dataloader, model, n_epochs, learning_rate=0.0003):
    """
    전체 학습 루프.

    Args:
        train_dataloader: 학습 데이터 로더
        model:            EncoderDecoder 모델
        n_epochs:         학습 에폭 수
        learning_rate:    Adam 옵티마이저 학습률
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # NLLLoss: Log-softmax 출력과 함께 사용하는 음 로그 가능도 손실
    # ignore_index=PAD_idx: 패딩 위치의 손실은 계산에서 제외
    criterion = nn.NLLLoss(ignore_index=PAD_idx)

    loss_history = []  # 에폭별 손실 기록

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        for iter, batch in enumerate(train_dataloader):
            # 배치 텐서 언팩: 모두 [B, SeqLen] shape
            input_tensor  = batch[0]  # 입력 시퀀스
            input_mask    = batch[1]  # 입력 마스크
            target_tensor = batch[2]  # 정답 시퀀스

            epoch_loss += train_step(
                input_tensor, input_mask, target_tensor,
                model, optimizer, criterion
            )

        avg_loss = epoch_loss / (iter + 1)
        loss_history.append(avg_loss)

        # 5 에폭마다 진행 상황 출력
        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{n_epochs}  Loss: {avg_loss:.4f}')

    return loss_history


def train_step(input_tensor, input_mask, target_tensor, model,
               optimizer, criterion):
    """
    단일 배치에 대한 순전파 + 역전파 + 파라미터 업데이트.

    Returns:
        float: 현재 배치의 스칼라 손실값
    """
    optimizer.zero_grad()   # 이전 그래디언트 초기화

    # 순전파: Teacher Forcing으로 target_tensor 전달
    decoder_outputs, _ = model(input_tensor, input_mask, target_tensor)

    # 손실 계산
    # decoder_outputs: [B, SeqLen, OutVocab] -> [B*SeqLen, OutVocab]
    # target_tensor:   [B, SeqLen]           -> [B*SeqLen]
    loss = criterion(
        decoder_outputs.view(-1, decoder_outputs.size(-1)),
        target_tensor.view(-1)
    )

    loss.backward()   # 역전파: 그래디언트 계산
    optimizer.step()  # 파라미터 업데이트

    return loss.item()


# ──────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────

def ids2words(lang, ids):
    """인덱스 리스트를 단어 리스트로 변환 (PAD/EOS 제외)"""
    words = []
    for idx in ids:
        if idx in (PAD_idx, EOS_token):
            break  # PAD나 EOS 만나면 중단
        word = lang.index2word.get(int(idx), '<UNK>')
        if word not in ('SOS', 'EOS'):
            words.append(word)
    return words


# ──────────────────────────────────────────
# BLEU 점수 계산 (간이 구현)
# ──────────────────────────────────────────

def compute_bleu(references, hypotheses, max_n=4):
    """
    BLEU-4 점수를 계산하는 간이 구현.

    BLEU(Bilingual Evaluation Understudy):
        번역 품질 자동 평가 지표. 1에 가까울수록 좋음.

    계산 방식:
        1. n-gram precision: 예측 문장의 n-gram이 정답에 얼마나 포함되는지
        2. Brevity Penalty (BP): 너무 짧은 번역에 패널티
        3. BLEU = BP × exp(Σ w_n × log(p_n))

    Args:
        references:  [[ref_words], ...]  - 정답 문장 리스트
        hypotheses:  [[hyp_words], ...]  - 예측 문장 리스트
        max_n:       최대 n-gram 크기 (보통 4)
    """
    # n별 precision 계산
    precisions = []
    for n in range(1, max_n + 1):
        match_count = 0
        total_count = 0

        for ref, hyp in zip(references, hypotheses):
            if len(hyp) < n:
                continue

            # 예측 문장의 n-gram 카운트
            hyp_ngrams = Counter(
                tuple(hyp[i:i+n]) for i in range(len(hyp) - n + 1)
            )
            # 정답 문장의 n-gram 카운트
            ref_ngrams = Counter(
                tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)
            )

            # 클리핑: 정답에 있는 만큼만 인정
            for ngram, cnt in hyp_ngrams.items():
                match_count += min(cnt, ref_ngrams.get(ngram, 0))
            total_count += sum(hyp_ngrams.values())

        if total_count == 0:
            precisions.append(0)
        else:
            precisions.append(match_count / total_count)

    # Brevity Penalty: 예측이 정답보다 짧으면 패널티
    ref_len = sum(len(r) for r in references)
    hyp_len = sum(len(h) for h in hypotheses)

    if hyp_len == 0:
        return 0.0

    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)

    # log precision 합산 (0 방지를 위해 smoothing)
    log_avg = sum(
        math.log(p + 1e-10) / max_n
        for p in precisions
    )

    return bp * math.exp(log_avg)


# ──────────────────────────────────────────
# 추론 및 성능 평가
# ──────────────────────────────────────────

def evaluate(model, dataloader, input_lang, output_lang, n_show=5):
    """
    학습된 모델로 번역을 수행하고 BLEU 점수를 계산.

    Args:
        n_show: 출력할 예시 문장 수
    """
    model.eval()   # 평가 모드: dropout, batchnorm 비활성화

    all_refs  = []  # 전체 정답 문장 수집
    all_hyps  = []  # 전체 예측 문장 수집

    with torch.no_grad():   # 그래디언트 계산 비활성화 (메모리/속도 절약)
        for batch_idx, batch in enumerate(dataloader):
            input_tensor  = batch[0]
            input_mask    = batch[1]
            target_tensor = batch[2]

            # 추론: target_tensor 없이 호출 → greedy decoding
            decoder_outputs, _ = model(input_tensor, input_mask)

            # 각 토큰 위치에서 가장 높은 확률의 인덱스 선택
            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze(-1)   # [B, SeqLen]

            for i in range(input_tensor.size(0)):
                hyp = ids2words(output_lang, decoded_ids[i].cpu().numpy())
                ref = ids2words(output_lang, target_tensor[i].cpu().numpy())
                all_refs.append(ref)
                all_hyps.append(hyp)

    # ── 예시 출력 ───────────────────────────
    print("\n" + "="*60)
    print("번역 예시 (Greedy Decoding)")
    print("="*60)
    for i in range(min(n_show, len(all_refs))):
        input_words = ids2words(input_lang, dataloader.dataset[i][0].cpu().numpy())
        print(f"  입력  : {' '.join(input_words)}")
        print(f"  정답  : {' '.join(all_refs[i])}")
        print(f"  예측  : {' '.join(all_hyps[i])}")
        print("-"*40)

    # ── BLEU 점수 계산 ───────────────────────
    bleu = compute_bleu(all_refs, all_hyps)

    # n-gram별 precision 계산 (추가 분석)
    print("\n" + "="*60)
    print("성능 평가")
    print("="*60)
    print(f"  BLEU-4 점수: {bleu:.4f}  ({bleu*100:.1f}/100)")

    # Exact Match: 완전히 일치하는 문장 비율
    exact = sum(1 for r, h in zip(all_refs, all_hyps) if r == h)
    print(f"  Exact Match: {exact}/{len(all_refs)} ({100*exact/len(all_refs):.1f}%)")

    model.train()   # 다시 학습 모드로 복귀
    return bleu


# ──────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────

if __name__ == '__main__':
    print("="*60)
    print("Seq2Seq + Bahdanau Attention 번역 모델")
    print(f"디바이스: {device}")
    print("="*60)

    # 1. 데이터 로드
    input_lang, output_lang, train_dataloader = load_data.get_dataloader(batch_size)

    # 2. 모델 초기화
    seq2seq = model.EncoderDecoder(
        hidden_size,
        input_lang.n_words,
        output_lang.n_words
    ).to(device)

    total_params = sum(p.numel() for p in seq2seq.parameters())
    print(f"\n모델 파라미터 수: {total_params:,}")
    print(f"  입력 어휘 크기: {input_lang.n_words}")
    print(f"  출력 어휘 크기: {output_lang.n_words}")
    print(f"  Hidden 차원:    {hidden_size}")

    # 3. 학습 전 기준 성능 (무작위 초기화 상태)
    print("\n[학습 전 성능]")
    evaluate(seq2seq, train_dataloader, input_lang, output_lang, n_show=3)

    # 4. 학습
    print("\n[학습 시작]")
    loss_history = train(train_dataloader, seq2seq, n_epochs=40)

    # 5. 학습 후 성능 평가
    print("\n[학습 후 성능]")
    final_bleu = evaluate(seq2seq, train_dataloader, input_lang, output_lang, n_show=5)

    print(f"\n최종 BLEU 점수: {final_bleu:.4f}")
