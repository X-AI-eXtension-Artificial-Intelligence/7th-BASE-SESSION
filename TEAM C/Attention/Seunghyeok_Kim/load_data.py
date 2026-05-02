from io import open
import unicodedata
import string
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10     # 문장 최대 길이 설정
SOS_token = 0       # SOS 토큰 할당
EOS_token = 1       # EOS 토큰 할당

#  영어 문장의 시작 부분에 자주 등장하는 구문들을 정의 -> 문장을 필터링할 때 사용

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

# Lang Class : 언어의 단어와 인덱스 매핑을 관리하는 클래스

class Lang:
    def __init__(self, name):       # 변수 초기화
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = { SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 2            # SOO 토큰 / EOS 토큰

    def addSentence(self, sentence):                        # 문장을 단어로 분리하여 addWord 메서드로 단어 추가
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):                                # 단어가 word2index에 없는 경우, 새로운 인덱스를 할당하고 단어 카운트를 1로 초기화. 이미 존재하는 경우, 단어 카운트 증가
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def filterPair(p):                                          # 문장 길이가 MAX_LENGTH보다 짧고, 영어 문장이 eng_prefixes 중 하나로 시작하는지 확인하여 True/False 반환
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):                                     # filterPair 함수를 사용하여 문장 쌍을 필터링하여 새로운 리스트 반환                                
    return [pair for pair in pairs if filterPair(pair)]


def unicodeToAscii(s):                                      # 유니코드 문자열을 ASCII로 변환하는 함수. NFD(Normalization Form Decomposed)로 정규화하여 발음 구별 기호를 분리한 후, 발음 구별 기호가 아닌 문자만 남김
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):                                     # 문자열을 소문자로 변환하고, 양쪽 공백 제거, 문장 부호 앞에 공백 추가, 알파벳과 문장 부호를 제외한 모든 문자를 공백으로 대체
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):                 # lang1과 lang2에 해당하는 언어의 문장 쌍을 읽어들이는 함수. reverse가 True인 경우, 문장 쌍을 뒤집어서 input_lang과 output_lang을 설정
    print("Reading lines...")

    # data/%s-%s.txt 파일을 읽어서 줄 단위로 분리하여 리스트로 저장. 각 줄은 lang1과 lang2의 문장 쌍으로 구성되어 있으며, 탭으로 구분되어 있음
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # 각 줄을 탭으로 분리하여 lang1과 lang2의 문장 쌍을 만들고, normalizeString 함수를 사용하여 각 문장을 정규화하여 pairs 리스트에 저장
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # 필요한 경우, 문장 쌍을 뒤집어서 input_lang과 output_lang을 설정 / reverse가 True인 경우, pairs의 각 요소를 뒤집어서 input_lang과 output_lang을 설정. reverse가 False인 경우, input_lang과 output_lang을 그대로 설정
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs




# prepareData 함수 : lang1과 lang2에 해당하는 언어의 문장 쌍을 읽어들이고, 필터링하여 input_lang과 output_lang 객체를 생성하고, pairs 리스트를 반환하는 함수
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

# indexesFromSentence 함수 : lang 객체와 문장을 입력으로 받아서, 문장을 단어로 분리하여 각 단어에 해당하는 인덱스를 리스트로 반환하는 함수
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

# tensorFromSentence 함수 : lang 객체와 문장을 입력으로 받아서, indexesFromSentence 함수를 사용하여 문장을 인덱스 리스트로 변환한 후, EOS_token을 추가하고, 이를 PyTorch 텐서로 변환하여 반환하는 함수
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# tensorsFromPair 함수 : lang 객체와 문장 쌍을 입력으로 받아서, tensorFromSentence 함수를 사용하여 input_lang과 output_lang에 해당하는 문장을 각각 텐서로 변환하여 반환하는 함수
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# get_dataloader 함수 : batch_size를 입력으로 받아서, prepareData 함수를 사용하여 input_lang, output_lang, pairs를 얻은 후, 각 문장 쌍을 인덱스 배열로 변환하여 input_ids, input_mask, target_ids, target_mask 배열을 생성. 이후, TensorDataset과 DataLoader를 사용하여 학습 데이터로 사용할 수 있는 train_dataloader를 반환하는 함수
def get_dataloader(batch_size):
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
