import math
from collections import Counter
import numpy as np

# BLEU score 계산을 위한 유틸리티 함수
def bleu_stats(hypothesis, reference):

    stats = []                              # BLEU 계산을 위한 통계 정보를 저장할 리스트
    stats.append(len(hypothesis))           # 가설 문장의 길이를 통계에 추가
    stats.append(len(reference))            # 참조 문장의 길이를 통계에 추가

    for n in range(1, 5):       # 1-gram부터 4-gram까지의 통계를 계산

        s_ngrams = Counter([tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)])        # 가설 문장에서 n-gram을 추출하여 카운터로 저장   
        r_ngrams = Counter([tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)])          # 참조 문장에서 n-gram을 추출하여 카운터로 저장

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))         # 가설과 참조에서 공통된 n-gram의 개수를 통계에 추가
        stats.append(max([len(hypothesis) + 1 - n, 0]))                     # 가설 문장에서 가능한 n-gram의 총 개수를 통계에 추가

    return stats

# BLEU 점수 계산 함수
def bleu(stats):

    if len(list(filter(lambda x: x == 0, stats))) > 0:      # 통계 정보 중에 0이 하나라도 있으면 BLEU 점수는 0이 됨
        return 0
    
    (c, r) = stats[:2]      # 가설 문장의 길이와 참조 문장의 길이를 가져옴
    log_bleu_prec = sum([math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]) / 4.        # n-gram 정밀도의 로그 평균을 계산하여 BLEU 점수의 일부로 사용
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)                                         # 길이 패널티와 n-gram 정밀도의 로그 평균을 합산하여 최종 BLEU 점수를 계산

# 가설 문장과 참조 문장 리스트를 입력으로 받아 BLEU 점수를 계산하는 함수
def get_bleu(hypotheses, reference):
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])      # BLEU 계산을 위한 초기 통계 정보 배열
    for hyp, ref in zip(hypotheses, reference):                     # 가설 문장과 참조 문장을 순회하면서 BLEU 통계 정보를 업데이트
        stats += np.array(bleu_stats(hyp, ref))                     # 각 가설과 참조 문장 쌍에 대해 BLEU 통계 정보를 계산하여 누적
    return 100 * bleu(stats)                                        # 최종 BLEU 점수를 100으로 스케일링하여 반환

# 인덱스 시퀀스를 단어 시퀀스로 변환하는 함수
def idx_to_word(x, vocab):
    words = []                      # 인덱스 시퀀스에서 단어 시퀀스로 변환된 결과를 저장할 리스트
    for i in x:                     # 인덱스 시퀀스의 각 인덱스에 대해 해당하는 단어를 찾아서 리스트에 추가
        word = vocab.itos[i]        # 인덱스에 해당하는 단어를 vocab의 itos(인덱스-단어 매핑)에서 가져옴
        if '<' not in word:         # 특수 토큰이 아닌 경우에만 단어 리스트에 추가
            words.append(word)      # 단어 리스트에 단어를 추가
    words = " ".join(words)         # 단어 리스트를 공백으로 구분된 문자열로 변환하여 반환
    return words
