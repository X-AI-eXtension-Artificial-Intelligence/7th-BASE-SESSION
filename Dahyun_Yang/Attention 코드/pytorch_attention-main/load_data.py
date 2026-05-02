from io import open
import unicodedata
import string
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# GPU 사용 가능 여부 확인 및 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10
SOS_token = 0 # Start of Sentence: 문장 시작
EOS_token = 1 # End of Sentence: 문장 끝

# 특정 패턴으로 시작하는 문장만 필터링하기 위한 접두사 리스트
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {} # 단어 -> 인덱스 맵핑
        self.word2count = {} # 단어 빈도
        self.index2word = { SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        # 문장을 받아 단어 단위로 쪼개어 사전에 추가
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            # 단어가 처음 등장하면 사전에 등록, 아니면 빈도수 증가
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def filterPair(p):
    # 문장 쌍이 최대 길이를 넘지 않고 특정 접두사로 시작하는지 검사
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    # 필터링된 문장 쌍 리스트 반환
    return [pair for pair in pairs if filterPair(pair)]

# Turn a Unicode string to plain ASCII, thanks to
# = 유니코드 문자를 아스키 코드로 변환 (악센트 제거 등)
# .
# 이 작업이 텍스트 표준화라는 작업임
# 단어 사전의 크기를 최적화하기 위해서 진행하는 작업
# 불어랑 영어를 비슷한 단어로 인식해서, 이러한 통일을 통해서 적은 데이터로도 단어의 의미를 효율적으로 학습
# 더불어 데이터 희소성 문제 해결(등장 빈도)
# 노이즈 제거 및 일관성 확보
# .
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
# 텍스트 전처리: 소문자화, 공백 제거, 특수문자 분리/제거
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs




######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#
def prepareData(lang1, lang2, reverse=False):
    # 데이터 전체 준비 프로세스: 로드 -> 필터링 -> 단어 사전 구축
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


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    # 문장을 모델 입력용 토치 텐서로 변환 (EOS 토큰 추가)
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def get_dataloader(batch_size):
    # 데이터셋을 배치 단위로 나누어 주는 DataLoader 생성
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    input_mask = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_mask = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        input_ids[idx, :len(inp_ids)] = inp_ids
        input_mask[idx, :len(inp_ids)] = 1
        target_ids[idx, :len(tgt_ids)] = tgt_ids
        target_mask[idx, :len(tgt_ids)] = 1

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(input_mask).to(device),
                               torch.LongTensor(target_ids).to(device),
                               torch.LongTensor(target_mask).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader
