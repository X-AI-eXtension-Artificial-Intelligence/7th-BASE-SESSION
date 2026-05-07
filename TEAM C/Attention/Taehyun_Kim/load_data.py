"""
load_data.py
============
영어-프랑스어 번역 데이터를 로드하고 전처리하는 모듈.
- 텍스트 정규화 (소문자, 특수문자 제거)
- 문장 길이/접두사 기준 필터링
- 단어 <-> 인덱스 매핑 (Lang 클래스)
- PyTorch DataLoader 생성
"""

from io import open
import unicodedata
import string
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# GPU가 있으면 CUDA 사용, 없으면 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 시퀀스 최대 길이: 이보다 긴 문장은 필터링됨
MAX_LENGTH = 10

# 특수 토큰 인덱스
SOS_token = 0   # Start Of Sentence: 디코더 입력의 시작 토큰
EOS_token = 1   # End Of Sentence: 문장 끝을 나타내는 토큰

# 영어 문장 필터링용 접두사 목록
# 이 접두사로 시작하는 영어 문장만 학습 데이터로 사용 (단순 패턴 집중)
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


# ──────────────────────────────────────────
# Lang 클래스: 단어 <-> 인덱스 양방향 매핑 관리
# ──────────────────────────────────────────
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}           # 단어 -> 인덱스 딕셔너리
        self.word2count = {}           # 단어 빈도 카운트
        self.index2word = {            # 인덱스 -> 단어 딕셔너리 (SOS, EOS 미리 등록)
            SOS_token: "SOS",
            EOS_token: "EOS"
        }
        self.n_words = 2               # 현재 등록된 단어 수 (SOS, EOS 포함해서 2부터 시작)

    def addSentence(self, sentence):
        """문장의 모든 단어를 어휘 사전에 추가"""
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """단어를 어휘 사전에 추가. 이미 있으면 빈도만 증가"""
        if word not in self.word2index:
            # 새 단어: 현재 n_words를 인덱스로 부여
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# ──────────────────────────────────────────
# 텍스트 정규화 함수들
# ──────────────────────────────────────────

def unicodeToAscii(s):
    """
    유니코드 문자열을 ASCII로 변환.
    예: 'é' -> 'e', 'ü' -> 'u'
    NFD 정규화 후 결합 문자(Mn 카테고리)를 제거하는 방식.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    """
    문자열 정규화 파이프라인:
    1. ASCII 변환 및 소문자화
    2. 구두점(.!?) 앞에 공백 삽입 → 별도 토큰으로 분리
    3. 알파벳/구두점 외 문자 제거
    예: "Où allez-vous?" -> "ou allez vous ?"
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)           # 구두점을 별도 토큰으로
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)       # 허용 문자 외 공백으로 치환
    return s


# ──────────────────────────────────────────
# 데이터 필터링 함수들
# ──────────────────────────────────────────

def filterPair(p):
    """
    단일 문장 쌍 필터링 조건:
    - 입력(프랑스어)과 출력(영어) 모두 MAX_LENGTH 미만
    - 출력(영어)이 eng_prefixes 중 하나로 시작
    (reverse=True이므로 p[1]이 영어)
    """
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    """전체 문장 쌍 리스트에서 조건을 만족하는 쌍만 반환"""
    return [pair for pair in pairs if filterPair(pair)]


# ──────────────────────────────────────────
# 파일 읽기 및 언어 객체 초기화
# ──────────────────────────────────────────

def readLangs(lang1, lang2, reverse=False):
    """
    탭 구분 번역 파일을 읽어 (입력 언어, 출력 언어, 문장쌍 리스트) 반환.

    reverse=False: lang1 -> lang2 방향 번역 (예: eng -> fra)
    reverse=True : lang2 -> lang1 방향 번역 (예: fra -> eng)
    """
    print("Reading lines...")

    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8') \
        .read().strip().split('\n')

    # 각 줄을 탭으로 분리 후 정규화
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:
        # 쌍 뒤집기: [fra, eng] 순서로
        pairs = [list(reversed(p)) for p in pairs]
        input_lang  = Lang(lang2)   # 입력: fra
        output_lang = Lang(lang1)   # 출력: eng
    else:
        input_lang  = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


# ──────────────────────────────────────────
# 데이터 준비 메인 함수
# ──────────────────────────────────────────

def prepareData(lang1, lang2, reverse=False):
    """
    전체 데이터 준비 파이프라인:
    1. 파일 읽기 및 문장 쌍 생성
    2. 길이/접두사 필터링
    3. 어휘 사전 구축
    """
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


# ──────────────────────────────────────────
# 텍스트 <-> 텐서 변환 유틸리티
# ──────────────────────────────────────────

def indexesFromSentence(lang, sentence):
    """문장의 각 단어를 어휘 인덱스로 변환 (리스트 반환)"""
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    """
    문장을 인덱스 텐서로 변환. EOS 토큰을 끝에 추가.
    반환 shape: [seq_len, 1]
    """
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    """문장 쌍을 (입력 텐서, 타겟 텐서) 튜플로 변환"""
    input_tensor  = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# ──────────────────────────────────────────
# DataLoader 생성
# ──────────────────────────────────────────

def get_dataloader(batch_size):
    """
    학습용 DataLoader를 생성하여 반환.

    모든 문장을 MAX_LENGTH 길이로 패딩(0으로):
    - input_ids  [N, MAX_LENGTH]: 입력 인덱스
    - input_mask [N, MAX_LENGTH]: 실제 토큰 위치 1, 패딩 위치 0
    - target_ids [N, MAX_LENGTH]: 타겟 인덱스
    - target_mask[N, MAX_LENGTH]: 실제 토큰 위치 1, 패딩 위치 0

    반환: (input_lang, output_lang, DataLoader)
    """
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    n = len(pairs)

    # 패딩 포함 배열 초기화 (기본값 0 = PAD)
    input_ids   = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    input_mask  = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids  = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_mask = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)

        # 실제 토큰 길이만큼 채우고, 나머지는 0(PAD)으로 유지
        input_ids  [idx, :len(inp_ids)] = inp_ids
        input_mask [idx, :len(inp_ids)] = 1
        target_ids [idx, :len(tgt_ids)] = tgt_ids
        target_mask[idx, :len(tgt_ids)] = 1

    # numpy 배열 -> PyTorch 텐서 -> TensorDataset
    train_data = TensorDataset(
        torch.LongTensor(input_ids ).to(device),
        torch.LongTensor(input_mask).to(device),
        torch.LongTensor(target_ids).to(device),
        torch.LongTensor(target_mask).to(device)
    )

    # RandomSampler: 매 에폭마다 무작위 순서로 배치 구성
    train_sampler    = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return input_lang, output_lang, train_dataloader
