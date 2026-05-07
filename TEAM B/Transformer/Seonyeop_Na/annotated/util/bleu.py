"""
util/bleu.py
- 번역 결과의 BLEU score를 직접 계산하는 파일입니다.
- BLEU는 n-gram overlap 기반의 기계번역 평가 지표입니다.
"""

import math
from collections import Counter

import numpy as np


def bleu_stats(hypothesis, reference):
    """한 문장 쌍에 대해 BLEU 계산에 필요한 통계를 만듭니다."""
    stats = []

    # 후보 번역 길이 c와 정답 번역 길이 r입니다.
    stats.append(len(hypothesis))
    stats.append(len(reference))

    # 1-gram부터 4-gram까지 overlap 수를 계산합니다.
    for n in range(1, 5):
        # hypothesis의 n-gram 빈도입니다.
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )

        # reference의 n-gram 빈도입니다.
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        # Counter의 & 연산은 두 Counter의 공통 원소에 대해 min count를 취합니다.
        # 즉, clipped n-gram match 개수입니다.
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))

        # 후보 문장에서 가능한 전체 n-gram 개수입니다.
        stats.append(max([len(hypothesis) + 1 - n, 0]))

    return stats


def bleu(stats):
    """누적 통계값으로 BLEU 점수를 계산합니다."""
    # 하나라도 0이 있으면 log 계산이 불가능하므로 0점을 반환합니다.
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0

    # c: 후보 번역 총 길이, r: 정답 번역 총 길이
    c, r = stats[:2]

    # 1~4 gram precision의 로그 평균입니다.
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.

    # brevity penalty + precision을 결합합니다.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """여러 문장에 대한 corpus-level BLEU를 계산합니다."""
    # [c, r, match1, total1, match2, total2, ... match4, total4]
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))

    # 일반적으로 BLEU를 0~100 스케일로 표시합니다.
    return 100 * bleu(stats)


def idx_to_word(x, vocab):
    """토큰 index 리스트를 실제 단어 문자열로 바꿉니다."""
    words = []

    for i in x:
        word = vocab.itos[i]

        # <pad>, <sos>, <eos>, <unk> 같은 특수 토큰은 제외합니다.
        if '<' not in word:
            words.append(word)

    words = " ".join(words)
    return words
