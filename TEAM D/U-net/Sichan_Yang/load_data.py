"""
load_data.py

Seq2Seq 기반 기계번역 모델 학습용 데이터 전처리 파일

이 파일의 핵심 목적

자연어 문장은 그대로 신경망에 입력 불가.
PyTorch 모델은 문자열이 아니라 정수 ID tensor를 입력으로 받음.
따라서 "문장 텍스트"를 "숫자 ID tensor"로 바꾸는 전처리 과정 필요.

전체 처리 흐름

1. data/eng-fra.txt 파일 읽기
   - 영어-프랑스어 문장쌍 데이터 사용
   - 번역 모델 학습에는 입력 문장과 정답 출력 문장 쌍 필요

2. 문장 정규화
   - 대소문자, 악센트, 특수문자 차이로 같은 단어가 다른 단어처럼 처리되는 문제 완화
   - 예: "Cold", "cold", "cold!" 같은 표현을 일관된 형태로 정리

3. 짧고 단순한 문장만 필터링
   - 실습용 작은 Seq2Seq 모델은 긴 문장 번역에 취약
   - 초반 학습 안정성을 위해 10단어 미만 문장만 사용
   - 특정 영어 prefix로 시작하는 쉬운 문장만 선택

4. 단어 사전 생성
   - 신경망은 단어 문자열을 직접 처리 불가
   - 각 단어를 고유한 정수 ID로 매핑 필요
   - 예: "je" → 5, "suis" → 12

5. 문장 → 정수 ID 배열 변환
   - embedding layer는 단어 ID를 입력으로 받음
   - 따라서 문장을 정수 ID sequence로 변환 필요

6. padding과 mask 생성
   - batch 학습을 하려면 한 batch 안의 문장 길이가 같아야 함
   - 짧은 문장은 0으로 채워 길이를 맞춤
   - attention이 padding 위치를 보지 않도록 mask 필요

7. TensorDataset과 DataLoader 생성
   - 학습 루프에서 batch 단위로 데이터를 쉽게 꺼내기 위한 구조
   - PyTorch 표준 학습 방식과 연결하기 위해 필요

주의점

- SOS_token = 0, padding 값도 0
  → 문장 시작 토큰과 padding 토큰 ID 충돌
  → 실전에서는 PAD=0, SOS=1, EOS=2처럼 분리 권장

- tensorFromSentence 함수에서는 EOS_token 추가
  하지만 get_dataloader에서는 indexesFromSentence만 사용
  → 실제 학습 데이터에는 EOS 토큰 미포함
  → decoder가 문장 종료 시점을 명시적으로 학습하기 어려움

- tensorsFromPair 함수는 input_lang, output_lang을 전역변수처럼 사용
  → 현재 흐름에서는 미사용
  → 직접 호출 시 NameError 가능

- SequentialSampler import 존재
  → 현재 코드에서는 미사용
"""


# 파일 입출력용 open 함수 import
# 텍스트 데이터 파일을 읽기 위해 필요
from io import open

# 유니코드 문자 정규화용 모듈 import
# 프랑스어 악센트 문자 제거를 위해 필요
import unicodedata

# 문자열 관련 모듈 import
# 현재 코드에서는 직접 사용 없음
# 원래 tutorial 코드에서 남은 import일 가능성
import string

# 정규표현식 처리용 모듈 import
# 문장부호 분리, 특수문자 제거 등 텍스트 정규화를 위해 필요
import re

# numpy 배열 처리용 모듈 import
# 고정 길이 input_ids, target_ids, mask 배열 생성을 위해 필요
import numpy as np

# PyTorch 메인 모듈 import
# tensor 생성, device 설정, 학습 데이터 변환을 위해 필요
import torch

# PyTorch Dataset, DataLoader, Sampler 관련 클래스 import
# TensorDataset: 여러 tensor를 하나의 dataset으로 묶기 위해 필요
# DataLoader: batch 단위 데이터 공급을 위해 필요
# RandomSampler: 학습 시 데이터 순서 섞기를 위해 필요
# SequentialSampler: 순차 샘플링용이나 현재 코드에서는 미사용
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# GPU 사용 가능 여부에 따른 device 설정
# 모델과 tensor가 같은 장치에 있어야 연산 가능
# GPU 사용 가능 시 학습 속도 향상 가능
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 최대 문장 길이 설정
# 너무 긴 문장을 제거하여 실습 모델의 학습 난이도 감소
# 고정 길이 배열 생성을 위한 기준 길이로도 사용
MAX_LENGTH = 10

# 문장 시작 토큰 ID
# decoder가 첫 단어 생성을 시작할 때 입력으로 사용
# 현재 코드에서는 padding ID와 겹치는 문제 존재
SOS_token = 0

# 문장 종료 토큰 ID
# 문장이 끝났음을 모델에 알려주기 위해 필요
# 단, 현재 get_dataloader에서는 실제로 붙이지 않음
EOS_token = 1


# 영어 출력 문장 필터링용 prefix 목록
# 쉬운 문장 구조만 남겨 Seq2Seq 모델이 안정적으로 학습하도록 하기 위한 제한
# 예: "i am tired", "he is good" 같은 짧고 단순한 문장 중심
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


# 언어별 단어 사전 관리 클래스
# 자연어 단어를 모델 입력용 정수 ID로 바꾸기 위해 필요
class Lang:

    # Lang 객체 초기화
    # 언어 이름과 단어 사전 자료구조 생성 목적
    def __init__(self, name):

        # 언어 이름 저장
        # 출력 로그 확인 및 언어 구분을 위해 필요
        self.name = name

        # 단어 → 정수 ID 사전
        # embedding layer 입력은 정수 ID이므로 단어를 숫자로 바꾸기 위해 필요
        self.word2index = {}

        # 단어 → 등장 횟수 사전
        # 단어 빈도 확인, rare word 처리 등에 활용 가능
        # 현재 코드에서는 카운트만 하고 직접 사용은 없음
        self.word2count = {}

        # 정수 ID → 단어 사전
        # 모델 예측 결과 ID를 다시 사람이 읽을 수 있는 단어로 복원하기 위해 필요
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}

        # 단어장 크기 초기값
        # SOS, EOS 두 특수 토큰을 이미 포함하므로 2부터 시작
        # 새로운 단어에 다음 ID를 부여하기 위해 필요
        self.n_words = 2  # Count SOS and EOS

    # 문장 단위 단어 추가
    # 문장 전체를 단어장에 등록하기 위해 필요
    def addSentence(self, sentence):

        # 공백 기준 단어 분리 후 단어장 추가
        # 현재 정규화 방식이 공백 기반 tokenization이므로 split 사용
        for word in sentence.split(' '):
            self.addWord(word)

    # 단어 단위 단어장 추가
    # 새 단어에는 ID 부여, 기존 단어는 빈도 증가
    def addWord(self, word):

        # 신규 단어인 경우
        # 처음 보는 단어에만 새 ID 부여 필요
        if word not in self.word2index:

            # 신규 단어에 현재 n_words 값 부여
            # 단어마다 고유 ID를 갖게 하기 위한 처리
            self.word2index[word] = self.n_words

            # 신규 단어 등장 횟수 1로 초기화
            # 이후 같은 단어가 나오면 count 증가
            self.word2count[word] = 1

            # 정수 ID → 단어 매핑 추가
            # decoding 결과를 단어로 되돌리기 위해 필요
            self.index2word[self.n_words] = word

            # 단어장 크기 1 증가
            # 다음 신규 단어가 다른 ID를 받도록 하기 위한 처리
            self.n_words += 1

        # 기존 단어인 경우
        # 새 ID를 만들 필요 없이 등장 횟수만 증가
        else:

            # 기존 단어 등장 횟수 1 증가
            # 단어 빈도 정보 유지 목적
            self.word2count[word] += 1


# 단일 문장쌍 필터링 조건 검사
# 학습하기 너무 어렵거나 긴 문장 제거 목적
def filterPair(p):

    # 입력 문장 길이 조건
    # 출력 문장 길이 조건
    # 출력 영어 문장 prefix 조건
    #
    # 필요한 이유:
    # - 긴 문장은 초보적인 Seq2Seq 모델에서 학습 난이도 증가
    # - 특정 prefix 문장만 남기면 문장 구조가 단순해져 실습 안정성 증가
    # - reverse=True 사용 시 p[1]은 영어 출력 문장
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


# 전체 문장쌍 리스트 필터링
# 데이터셋 전체에서 학습 조건에 맞는 문장쌍만 선택하기 위해 필요
def filterPairs(pairs):

    # filterPair 조건 만족 문장쌍만 선택
    # 불필요하거나 어려운 샘플 제거 목적
    return [pair for pair in pairs if filterPair(pair)]


# 유니코드 문자열 → ASCII 문자열 변환
# 악센트 차이로 같은 단어가 다른 단어처럼 처리되는 문제 완화 목적
def unicodeToAscii(s):

    # NFD 정규화 후 결합 문자 제거
    # 예: é → e + 악센트 기호 분해 후 악센트 제거
    # vocabulary 크기 증가 억제 및 텍스트 표준화 목적
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# 문장 정규화 함수
# 원본 문장의 표기 차이를 줄이고 tokenization을 단순화하기 위해 필요
def normalizeString(s):

    # 소문자 변환, 양끝 공백 제거, 악센트 제거
    # "I", "i", "é", "e" 같은 표기 차이 정리 목적
    s = unicodeToAscii(s.lower().strip())

    # 문장부호 앞 공백 추가
    # "cold!"를 "cold !"처럼 분리하여 문장부호도 별도 token처럼 처리하기 위함
    s = re.sub(r"([.!?])", r" \1", s)

    # 알파벳과 .!? 외 문자 공백 치환
    # 모델이 다루는 문자 범위를 제한하여 vocabulary 복잡도 감소
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    # 정규화 문자열 반환
    # 이후 단어 사전 생성 및 ID 변환에 사용
    return s


# 데이터 파일 읽기 및 문장쌍 생성 함수
# 원본 텍스트 파일을 모델 학습용 문장쌍 리스트로 바꾸기 위해 필요
def readLangs(lang1, lang2, reverse=False):

    # 파일 읽기 시작 메시지 출력
    # 데이터 로딩 진행 상태 확인 목적
    print("Reading lines...")

    # data/{lang1}-{lang2}.txt 파일 읽기
    # 전체 텍스트를 줄 단위 리스트로 분리
    #
    # 필요한 이유:
    # - 번역 학습은 문장쌍 단위로 진행
    # - 한 줄이 하나의 문장쌍이라고 가정
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # 각 줄을 탭 기준으로 분리 후 문장 정규화
    #
    # 필요한 이유:
    # - 원본 줄: "영어문장\t프랑스어문장"
    # - 모델 학습에는 [입력문장, 출력문장] pair 구조 필요
    # - 정규화된 문장만 단어 사전과 tensor 변환에 사용 가능
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # 번역 방향 반전 옵션
    # 원본 데이터 방향과 원하는 번역 방향이 다를 수 있기 때문에 필요
    if reverse:

        # 각 문장쌍 순서 반전
        # 예: [영어, 프랑스어] → [프랑스어, 영어]
        pairs = [list(reversed(p)) for p in pairs]

        # 입력 언어 객체 생성
        # reverse=True이므로 원래 lang2가 입력 언어
        input_lang = Lang(lang2)

        # 출력 언어 객체 생성
        # reverse=True이므로 원래 lang1이 출력 언어
        output_lang = Lang(lang1)

    # 번역 방향 유지 옵션
    # 원본 파일의 방향 그대로 학습할 때 필요
    else:

        # 입력 언어 객체 생성
        input_lang = Lang(lang1)

        # 출력 언어 객체 생성
        output_lang = Lang(lang2)

    # 입력 언어, 출력 언어, 문장쌍 반환
    # 이후 prepareData에서 필터링과 단어 사전 생성에 사용
    return input_lang, output_lang, pairs


######################################################################
# 데이터 준비 전체 과정
#
# 1. 텍스트 파일 읽기
#    - 원본 번역 데이터 확보 목적
#
# 2. 줄 단위 분리
#    - 한 줄을 하나의 문장쌍으로 처리하기 위함
#
# 3. 입력 문장과 출력 문장 분리
#    - supervised translation 학습에는 source-target pair 필요
#
# 4. 문장 정규화
#    - vocabulary 복잡도 감소 및 표기 일관성 확보 목적
#
# 5. 길이와 prefix 조건 기반 문장쌍 필터링
#    - 실습 모델이 감당 가능한 쉬운 샘플만 선택
#
# 6. 단어 사전 생성
#    - 문자열 단어를 embedding layer 입력용 정수 ID로 변환하기 위함
######################################################################


# 전체 데이터 전처리 함수
# 파일 읽기, 필터링, 단어 사전 생성을 한 번에 수행하기 위해 필요
def prepareData(lang1, lang2, reverse=False):

    # 파일 읽기 및 문장쌍 생성
    # 원본 텍스트를 정규화된 pair 리스트로 변환
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)

    # 필터링 전 문장쌍 개수 출력
    # 원본 데이터 규모 확인 목적
    print("Read %s sentence pairs" % len(pairs))

    # 문장쌍 필터링
    # 긴 문장과 복잡한 출력 구조 제거 목적
    pairs = filterPairs(pairs)

    # 필터링 후 문장쌍 개수 출력
    # 실제 학습에 쓰이는 데이터 규모 확인 목적
    print("Trimmed to %s sentence pairs" % len(pairs))

    # 단어 사전 생성 시작 메시지 출력
    # 전처리 진행 상태 확인 목적
    print("Counting words...")

    # 모든 문장쌍 순회
    # 입력 언어와 출력 언어 각각의 vocabulary 구축 목적
    for pair in pairs:

        # 입력 문장 단어장 추가
        # encoder embedding 입력 vocabulary 생성 목적
        input_lang.addSentence(pair[0])

        # 출력 문장 단어장 추가
        # decoder embedding 및 출력 softmax vocabulary 생성 목적
        output_lang.addSentence(pair[1])

    # 단어 사전 생성 완료 메시지 출력
    print("Counted words:")

    # 입력 언어명과 입력 단어장 크기 출력
    # encoder input vocabulary 크기 확인 목적
    print(input_lang.name, input_lang.n_words)

    # 출력 언어명과 출력 단어장 크기 출력
    # decoder output vocabulary 크기 확인 목적
    print(output_lang.name, output_lang.n_words)

    # 입력 언어, 출력 언어, 문장쌍 반환
    # 이후 DataLoader 생성과 모델 vocabulary size 설정에 필요
    return input_lang, output_lang, pairs


# 문장 → 단어 ID 리스트 변환 함수
# embedding layer에 넣기 위해 문자열 단어를 정수 ID로 변환 필요
def indexesFromSentence(lang, sentence):

    # 공백 기준 단어 분리 후 각 단어를 정수 ID로 변환
    # 예: "i am cold" → [2, 3, 10]
    # 신경망은 문자열이 아니라 정수 ID를 입력으로 받기 때문에 필요
    return [lang.word2index[word] for word in sentence.split(' ')]


# 문장 → tensor 변환 함수
# 단일 문장을 PyTorch tensor 형태로 변환하기 위해 필요
def tensorFromSentence(lang, sentence):

    # 문장 → 단어 ID 리스트 변환
    # 문자열 문장을 수치 sequence로 변환
    indexes = indexesFromSentence(lang, sentence)

    # 문장 끝에 EOS 토큰 추가
    # decoder가 문장 종료 시점을 학습할 수 있게 하기 위한 처리
    # 단, get_dataloader에서는 이 함수 미사용
    indexes.append(EOS_token)

    # 단어 ID 리스트 → LongTensor 변환 후 [길이, 1] 형태 변경
    #
    # 필요한 이유:
    # - embedding layer는 LongTensor 타입의 token ID 입력 필요
    # - device 이동으로 모델과 같은 장치에서 연산 가능
    # - view(-1, 1)은 기존 tutorial의 비배치 단일 문장 처리 형식
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


# 문장쌍 → 입력 tensor, target tensor 변환 함수
# 단일 pair를 모델 학습 입력 형태로 바꾸기 위한 함수
# 현재 get_dataloader 방식에서는 사용되지 않음
def tensorsFromPair(pair):

    # 입력 문장 tensor 변환
    # encoder 입력으로 사용하기 위한 변환
    # 주의: input_lang이 함수 인자로 전달되지 않음
    input_tensor = tensorFromSentence(input_lang, pair[0])

    # 출력 문장 tensor 변환
    # decoder target으로 사용하기 위한 변환
    # 주의: output_lang이 함수 인자로 전달되지 않음
    target_tensor = tensorFromSentence(output_lang, pair[1])

    # 입력 tensor와 target tensor 반환
    # 학습 루프에서 source-target pair로 사용 가능
    return (input_tensor, target_tensor)


# 학습용 DataLoader 생성 함수
# 전체 데이터 전처리 결과를 batch 학습 가능한 PyTorch DataLoader로 변환하기 위해 필요
def get_dataloader(batch_size):

    # eng-fra 데이터 읽기
    # reverse=True이므로 실제 번역 방향은 fra → eng
    #
    # 필요한 이유:
    # - encoder 입력: 프랑스어 문장
    # - decoder 출력/target: 영어 문장
    # - input_lang, output_lang은 모델 vocabulary size 설정에도 사용
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    # 문장쌍 개수 저장
    # numpy 배열의 첫 번째 차원 크기 결정에 필요
    n = len(pairs)

    # 입력 문장 ID 배열 초기화
    #
    # shape: [문장쌍 개수, 최대 문장 길이]
    # 필요한 이유:
    # - batch 학습을 위해 모든 입력 문장의 길이를 MAX_LENGTH로 통일
    # - 0으로 초기화하여 짧은 문장의 뒷부분을 padding 처리
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    # 입력 문장 mask 배열 초기화
    #
    # 필요한 이유:
    # - attention 계산 시 padding 위치를 무시해야 함
    # - 실제 단어 위치는 1, padding 위치는 0
    input_mask = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    # 출력 문장 ID 배열 초기화
    #
    # 필요한 이유:
    # - decoder target도 batch 학습을 위해 고정 길이 필요
    # - NLLLoss 계산 시 target token ID로 사용
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    # 출력 문장 mask 배열 초기화
    #
    # 필요한 이유:
    # - target padding 위치 구분 가능
    # - 현재 train.py에서는 직접 사용하지 않음
    # - 대신 NLLLoss(ignore_index=PAD_idx)로 padding 무시
    target_mask = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    # 모든 문장쌍 순회
    # 각 문장을 정수 ID 배열과 mask 배열로 채우기 위한 반복
    for idx, (inp, tgt) in enumerate(pairs):

        # 입력 문장 → 단어 ID 리스트 변환
        # encoder embedding 입력 생성 목적
        inp_ids = indexesFromSentence(input_lang, inp)

        # 출력 문장 → 단어 ID 리스트 변환
        # decoder target 생성 목적
        tgt_ids = indexesFromSentence(output_lang, tgt)

        # 입력 ID 배열에 실제 단어 ID 기록
        #
        # 필요한 이유:
        # - 각 문장을 MAX_LENGTH 길이 배열의 앞부분에 배치
        # - 나머지 위치는 0 padding 유지
        input_ids[idx, :len(inp_ids)] = inp_ids

        # 입력 mask의 실제 단어 위치 1로 기록
        #
        # 필요한 이유:
        # - attention에서 실제 단어에는 가중치 부여 가능
        # - padding 위치에는 attention이 가지 않도록 표시
        input_mask[idx, :len(inp_ids)] = 1

        # 출력 ID 배열에 실제 단어 ID 기록
        #
        # 필요한 이유:
        # - loss 계산 시 정답 target token으로 사용
        # - 나머지 위치는 0 padding 유지
        target_ids[idx, :len(tgt_ids)] = tgt_ids

        # 출력 mask의 실제 단어 위치 1로 기록
        #
        # 필요한 이유:
        # - target padding 위치 구분 가능
        # - 현재 코드에서는 TensorDataset에 포함되지만 train.py에서 직접 활용은 제한적
        target_mask[idx, :len(tgt_ids)] = 1

    # numpy 배열 → PyTorch LongTensor 변환
    # TensorDataset으로 입력, 입력 mask, target, target mask 묶음
    #
    # 필요한 이유:
    # - PyTorch 모델은 torch.Tensor 입력 필요
    # - embedding layer는 LongTensor token ID 필요
    # - TensorDataset은 여러 tensor를 sample 단위로 묶어 DataLoader에 전달 가능
    # - device 이동으로 모델과 같은 장치에서 연산 가능
    train_data = TensorDataset(
        torch.LongTensor(input_ids).to(device),
        torch.LongTensor(input_mask).to(device),
        torch.LongTensor(target_ids).to(device),
        torch.LongTensor(target_mask).to(device)
    )

    # 무작위 샘플링용 sampler 생성
    #
    # 필요한 이유:
    # - 매 epoch마다 데이터 순서를 섞으면 특정 순서에 대한 편향 감소
    # - mini-batch SGD/Adam 학습 안정성에 일반적으로 유리
    train_sampler = RandomSampler(train_data)

    # batch 단위 데이터 로딩용 DataLoader 생성
    #
    # 필요한 이유:
    # - 전체 데이터를 한 번에 학습하지 않고 batch 단위로 나누어 학습
    # - 메모리 사용량 제어
    # - optimizer update를 batch 단위로 수행 가능
    train_dataloader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=batch_size
    )

    # 입력 언어, 출력 언어, DataLoader 반환
    #
    # 필요한 이유:
    # - input_lang.n_words: encoder vocabulary size 설정에 필요
    # - output_lang.n_words: decoder vocabulary size 설정에 필요
    # - train_dataloader: 실제 학습 loop에서 batch 공급
    return input_lang, output_lang, train_dataloader