from io import open
import unicodedata
import string
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 설정값 ---
MAX_LENGTH = 10 # 번역할 문장의 최대 길이
SOS_token = 0   # 문장의 시작(Start of Sentence) 토큰
EOS_token = 1   # 문장의 끝(End of Sentence) 토큰

# 학습 속도를 높이기 위해 특정 접두사로 시작하는 짧은 문장만 필터링하기 위한 튜플
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

# --- 어휘 사전 클래스 ---
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {} # 단어 -> 인덱스 매핑
        self.word2count = {} # 단어의 출현 빈도수
        self.index2word = { SOS_token: "SOS", EOS_token: "EOS"} # 인덱스 -> 단어 매핑
        self.n_words = 2  # 어휘 사전의 전체 단어 수 (SOS와 EOS를 포함해 2부터 시작)

    def addSentence(self, sentence):
        # 문장을 공백 기준으로 나누어 사전에 추가
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        # 새로운 단어면 사전에 등록하고, 이미 있으면 빈도수만 증가
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# --- 데이터 필터링 함수 ---
def filterPair(p):
    # 양쪽 언어 문장의 길이가 MAX_LENGTH 미만이고, 영어 문장이 지정된 접두사로 시작하는지 확인
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# --- 텍스트 정규화 함수 ---
def unicodeToAscii(s):
    # 유니코드 문자열을 일반 ASCII로 변환
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    # 소문자 변환, 양쪽 공백 제거, 구두점 띄어쓰기 처리, 영문 및 구두점 외 문자 제거
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# --- 파일 읽기 함수 ---
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    # 텍스트 파일을 줄 단위로 읽어옴
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # 각 줄을 탭(\t) 기준으로 분리하여 (입력문, 출력문) 쌍 생성 후 정규화
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # reverse 플래그가 True면 타겟 언어에서 소스 언어로 번역하도록 언어 쌍을 뒤집음
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

# --- 전체 데이터 준비 파이프라인 ---
def prepareData(lang1, lang2, reverse=False):
    # 1. 파일 읽기
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    # 2. 조건에 맞는 쌍만 필터링
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    # 3. 언어별 어휘 사전 구축
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

# --- 문장을 텐서로 변환하는 유틸리티 함수들 ---
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token) # 문장 끝에 EOS 토큰 추가
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# --- DataLoader 생성기 ---
def get_dataloader(batch_size):
    # 데이터 준비 (영어를 프랑스어로 번역하는 역방향 설정)
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    n = len(pairs)
    # 배치 처리를 위해 고정 길이(MAX_LENGTH) 배열을 0(Padding)으로 초기화
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    input_mask = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_mask = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    # 데이터 배열 채우기
    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        
        input_ids[idx, :len(inp_ids)] = inp_ids
        input_mask[idx, :len(inp_ids)] = 1 # 실제 데이터가 있는 곳은 1로 마스킹
        target_ids[idx, :len(tgt_ids)] = tgt_ids
        target_mask[idx, :len(tgt_ids)] = 1

    # PyTorch TensorDataset으로 묶음
    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(input_mask).to(device),
                               torch.LongTensor(target_ids).to(device),
                               torch.LongTensor(target_mask).to(device))

    # 랜덤 샘플링하는 DataLoader 생성 반환
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader