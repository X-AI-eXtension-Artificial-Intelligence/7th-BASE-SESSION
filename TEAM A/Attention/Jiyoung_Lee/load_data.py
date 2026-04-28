import torch
from torch.utils.data import DataLoader, Dataset

MAX_LENGTH = 10

class Lang:
    def __init__(self):
        self.word2index = {"<pad>":0, "<sos>":1, "<eos>":2}
        self.index2word = {0:"<pad>", 1:"<sos>", 2:"<eos>"}
        self.n_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def read_data(path):
    pairs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            eng, fra = line.strip().split('\t')
            pairs.append((eng.lower(), fra.lower()))
    return pairs


def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def prepare_data(path):
    pairs = read_data(path)
    pairs = [p for p in pairs if filter_pair(p)]

    input_lang = Lang()
    output_lang = Lang()

    for p in pairs:
        input_lang.add_sentence(p[0])
        output_lang.add_sentence(p[1])

    return input_lang, output_lang, pairs


def tensor_from_sentence(lang, sentence):
    indexes = [lang.word2index[w] for w in sentence.split(' ')]
    indexes.append(2)  # EOS

    # padding
    if len(indexes) < MAX_LENGTH:
        indexes += [0] * (MAX_LENGTH - len(indexes))
    else:
        indexes = indexes[:MAX_LENGTH]

    return torch.tensor(indexes, dtype=torch.long)


class TranslationDataset(Dataset):
    def __init__(self, pairs, input_lang, output_lang):
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, trg = self.pairs[idx]
        return tensor_from_sentence(self.input_lang, src), tensor_from_sentence(self.output_lang, trg)


def get_dataloader(data_path, batch_size=1):
    input_lang, output_lang, pairs = prepare_data(data_path)
    dataset = TranslationDataset(pairs, input_lang, output_lang)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return input_lang, output_lang, loader