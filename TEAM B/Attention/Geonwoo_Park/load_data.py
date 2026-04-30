from io import open
import unicodedata
import string
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10        # toy setting. 논문은 30~50
SOS_token = 0          # 주의: train.py의 PAD_idx도 0. 같은 값 재사용 중
EOS_token = 1

# 영어 prefix 필터 - 인칭대명사+be 패턴만 학습
# 사실상 단순 문장만 보겠다는 의미
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


class Lang:
    """언어별 vocabulary 관리. word <-> id 양방향 사전."""
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}                                       # 빈도 추적은 하지만 실제로 사용 X
        self.index2word = { SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 2  # SOS, EOS 예약

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def filterPair(p):
    # 양쪽 다 MAX_LENGTH 미만 + 영어가 정해진 prefix로 시작
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def unicodeToAscii(s):
    # café -> cafe. 프랑스어 액센트 정보 손실됨 (toy니까 OK)
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)        # 구두점을 별도 토큰으로 분리
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)    # 알파벳/구두점 외 모두 제거 (숫자, 특수문자 다 날아감)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # reverse=True면 fr->en이 아니라 en->fr이 됨 (파일 자체는 en\tfr 순서)
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareData(lang1, lang2, reverse=False):
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
    # OOV 핸들링 없음. 학습셋 외 단어 들어오면 KeyError
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    n = len(pairs)
    # 모두 MAX_LENGTH 길이 텐서로 미리 만들고 padding 위치는 0으로 둠
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    input_mask = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_mask = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        # 실제 단어 위치만 채우고, 그 위치들의 mask=1
        # mask는 attention에서 padding 무시하는 데 사용 (model.py의 masked_fill)
        input_ids[idx, :len(inp_ids)] = inp_ids
        input_mask[idx, :len(inp_ids)] = 1
        target_ids[idx, :len(tgt_ids)] = tgt_ids
        target_mask[idx, :len(tgt_ids)] = 1
        # NOTE: target에 EOS는 안 붙음 (indexesFromSentence가 EOS 안 추가)
        # tensorFromSentence에서는 붙는데 여기서는 indexesFromSentence 직접 호출

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(input_mask).to(device),
                               torch.LongTensor(target_ids).to(device),
                               torch.LongTensor(target_mask).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader
