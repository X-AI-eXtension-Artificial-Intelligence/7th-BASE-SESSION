from io import open
import unicodedata
import string
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# GPU를 사용할 수 있으면 GPU를, 아니면 CPU를 사용합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 패딩용 고정 시퀀스 길이입니다.
MAX_LENGTH = 10  # 10은 문장의 최대 길이를 10으로 제한한다는 뜻입
# 단어 사전에서 사용하는 특수 토큰 ID입니다.
SOS_token = 0 # 문장 시작 토큰
EOS_token = 1 # 문장 종료 토큰  

# 단순한 학습용 부분집합을 만들기 위해 남길 영어 문장 시작 패턴입니다.
# 전체 영어-프랑스어 데이터에는 다양한 문장이 들어있을 수 있습니다.
# 그중에서 영어 문장이 특정 패턴으로 시작하는 것만 남깁니다.
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


class Lang:
    """한 언어의 단어 사전을 담는 간단한 클래스."""
    def __init__(self, name):
        self.name = name 
        # 단어 -> 정수 ID : 단어를 숫자 ID로 바꿔주는 딕셔너리
        self.word2index = {}
        # 단어 -> 등장 횟수
        self.word2count = {}
        # 정수 ID -> 단어
        self.index2word = { SOS_token: "SOS", EOS_token: "EOS"}
        # SOS/EOS를 미리 예약해 두고 시작합니다.
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        # 문장 하나를 받아서 공백 기준으로 단어를 나눈 뒤, 각 단어를 사전에 추가
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        # 처음 본 단어면 사전에 추가하고, 이미 있으면 빈도만 증가시킵니다.
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def filterPair(p):
    # 입력문장 길이가 10보다 짧은 문장쌍만 남기고, 타깃 영어 문장은 eng_prefixes에 있는 패턴으로 시작할 때만 유지합니다.
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    # 전체 문장쌍에 filterPair를 적용합니다.
    return [pair for pair in pairs if filterPair(pair)]

# 유니코드 문자열을 ASCII로 바꿉니다.
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    # 프랑스어의 악센트 문자를 분해한 뒤 결합 기호(발음 부호)를 제거합니다.
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 소문자화, 공백 정리, 비문자 제거를 수행합니다.
def normalizeString(s):
    # 1) 소문자 변환 + 앞뒤 공백 제거
    s = unicodeToAscii(s.lower().strip())
    # 2) 문장부호 앞에 공백을 넣어 토큰으로 분리되게 함
    s = re.sub(r"([.!?])", r" \1", s)
    # 3) 영문자와 . ! ? 이외 문자는 공백으로 치환
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# 데이터 파일을 줄 단위로 나눈 뒤, 각 줄을 문장쌍으로 분리해 읽습니다.
# 원본 파일은 English -> Other Language 방향이므로,
# Other Language -> English로 학습하고 싶을 때 reverse 플래그로 순서를 뒤집습니다.
def readLangs(lang1, lang2, reverse=False): # 데이터 파일을 읽고 문장쌍을 만드는 함수
    print("Reading lines...")

    # 탭으로 구분된 병렬 문장 파일을 읽어 줄 단위로 분리합니다.
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # 각 줄을 문장쌍으로 분해하고 양쪽 문장을 정규화합니다.
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # 필요하면 번역 방향(입력/출력 언어)을 뒤집습니다.
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs
# I am hungry.    Je suis faim. => [["i am hungry .", "je suis faim ."]]



######################################################################
# 데이터 준비 전체 과정은 다음과 같습니다.
#
# - 텍스트 파일 읽기 -> 줄 단위 분리 -> 문장쌍 분리
# - 텍스트 정규화 및 길이/패턴 필터링
# - 문장쌍으로부터 단어 사전 구축
# 1. 파일 읽기 -> 2. 문장 정규화 -> 3. 문장쌍 필터링 -> 4. 단어 사전 만들기 -> 5. input_lang, output_lang, pairs 반환

def prepareData(lang1, lang2, reverse=False):
    # 원시 데이터를 읽고 정규화합니다.
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    # 짧고 단순한 패턴의 문장만 남깁니다.
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    # 필터링된 데이터로 입력/출력 단어 사전을 만듭니다.
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    # 문장을 토큰 ID 시퀀스로 변환합니다.
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    # 문장 끝 학습을 위해 EOS 토큰을 붙입니다.
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    # 튜토리얼 스타일에 맞춰 [-1, 1] 형태 텐서로 변환합니다.
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    # (입력, 타깃) 문장쌍 1개를 텐서로 변환합니다.
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def get_dataloader(batch_size):
    # reverse=True 이므로 입력은 프랑스어, 출력은 영어입니다.
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    n = len(pairs)
    # 고정 길이 배열을 미리 생성합니다.
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    input_mask = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_mask = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        # 문장을 토큰 ID로 변환합니다. (이 경로에서는 EOS를 따로 붙이지 않음)
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        # 왼쪽부터 채우고, 남는 칸은 0으로 패딩됩니다.
        input_ids[idx, :len(inp_ids)] = inp_ids
        input_mask[idx, :len(inp_ids)] = 1
        target_ids[idx, :len(tgt_ids)] = tgt_ids
        target_mask[idx, :len(tgt_ids)] = 1

    # 텐서를 선택된 디바이스(CPU/GPU)로 이동합니다.
    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(input_mask).to(device),
                               torch.LongTensor(target_ids).to(device),
                               torch.LongTensor(target_mask).to(device))

    # 학습 시 배치를 무작위 샘플링합니다.
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader
