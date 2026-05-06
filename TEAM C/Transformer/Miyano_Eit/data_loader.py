import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build(self, sentences, min_freq=1):
        counter = Counter(
            word for sentence in sentences for word in sentence.split()
        )
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence, max_len):
        tokens = sentence.split()
        ids = [self.word2idx.get(t, 3) for t in tokens]
        ids = [1] + ids + [2]
        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=50):
        self.src = [src_vocab.encode(s, max_len) for s in src_sentences]
        self.tgt = [tgt_vocab.encode(t, max_len) for t in tgt_sentences]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src[idx], dtype=torch.long),
            torch.tensor(self.tgt[idx], dtype=torch.long),
        )


def get_dataloader(src_sentences, tgt_sentences, src_vocab, tgt_vocab,
                   max_len=50, batch_size=32, shuffle=True):
    dataset = TranslationDataset(src_sentences, tgt_sentences,
                                 src_vocab, tgt_vocab, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)