"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import gzip
import os
import urllib.request
from collections import namedtuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

Batch = namedtuple("Batch", ["src", "trg"])

# Official Multi30k raw files (Sheffield URLs used by torchtext are often broken).
_MULTI30K_BASE = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
_SPLIT_GZ = {
    "train": ("train.en.gz", "train.de.gz"),
    "valid": ("val.en.gz", "val.de.gz"),
    "test": ("test_2016_flickr.en.gz", "test_2016_flickr.de.gz"),
}


def _attach_legacy_vocab_attrs(vocab):
    """BLEU helper expects .itos (list) and .stoi (dict)."""
    vocab.itos = vocab.get_itos()
    vocab.stoi = vocab.get_stoi()
    return vocab


def _read_cached_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [line.strip() for line in f]


def _ensure_multi30k_split(cache_dir, split):
    """Download gz pair once, return list of (src_line, trg_line) for EN–DE."""
    os.makedirs(cache_dir, exist_ok=True)
    en_name, de_name = _SPLIT_GZ[split]
    paths = []
    for name in (en_name, de_name):
        local = os.path.join(cache_dir, name)
        if not os.path.isfile(local):
            url = _MULTI30K_BASE + name
            print(f"downloading {name} …")
            tmp = local + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, local)
        paths.append(local)
    en_lines = _read_cached_gz(paths[0])
    de_lines = _read_cached_gz(paths[1])
    if len(en_lines) != len(de_lines):
        raise ValueError(f"{split}: line count mismatch {len(en_lines)} vs {len(de_lines)}")
    return list(zip(en_lines, de_lines))


def _pairs_for_language_order(pairs_en_de, src_lang, trg_lang):
    if (src_lang, trg_lang) == ("en", "de"):
        return pairs_en_de
    if (src_lang, trg_lang) == ("de", "en"):
        return [(de, en) for en, de in pairs_en_de]
    raise ValueError(f"unsupported pair {(src_lang, trg_lang)}")


class _TranslationDataset(Dataset):
    def __init__(self, pairs, src_tokenize, trg_tokenize, src_vocab, trg_vocab):
        self.pairs = pairs
        self.src_tokenize = src_tokenize
        self.trg_tokenize = trg_tokenize
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, trg_text = self.pairs[idx]
        src_toks = self.src_tokenize(src_text.lower())
        trg_toks = self.trg_tokenize(trg_text.lower())
        src_ids = [self.src_vocab["<sos>"]] + [self.src_vocab[t] for t in src_toks] + [self.src_vocab["<eos>"]]
        trg_ids = [self.trg_vocab["<sos>"]] + [self.trg_vocab[t] for t in trg_toks] + [self.trg_vocab["<eos>"]]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)


def _collate_batch(pad_idx, device):
    def _collate(batch):
        src_batch, trg_batch = zip(*batch)
        src = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
        trg = pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx)
        return Batch(src=src.to(device), trg=trg.to(device))

    return _collate


class DataLoader:
    source = None
    target = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        self._data_root = os.path.join(os.getcwd(), ".data")
        os.makedirs(self._data_root, exist_ok=True)
        print("dataset initializing start")

    def make_dataset(self):
        if self.ext == (".de", ".en"):
            language_pair = ("de", "en")
            src_tok = self.tokenize_de
            trg_tok = self.tokenize_en
        elif self.ext == (".en", ".de"):
            language_pair = ("en", "de")
            src_tok = self.tokenize_en
            trg_tok = self.tokenize_de
        else:
            raise ValueError(f"unsupported ext {self.ext}")

        self._src_tok = src_tok
        self._trg_tok = trg_tok
        self._language_pair = language_pair

        cache_dir = os.path.join(self._data_root, "multi30k")

        def _materialize(split):
            pairs = _ensure_multi30k_split(cache_dir, split)
            return _pairs_for_language_order(pairs, language_pair[0], language_pair[1])

        train_data = _materialize("train")
        valid_data = _materialize("valid")
        test_data = _materialize("test")
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        specials = ["<unk>", "<pad>", "<sos>", "<eos>"]

        def yield_src():
            for src, _ in train_data:
                yield self._src_tok(src.lower())

        def yield_trg():
            for _, trg in train_data:
                yield self._trg_tok(trg.lower())

        src_vocab = build_vocab_from_iterator(
            yield_src(), min_freq=min_freq, specials=specials, special_first=True
        )
        trg_vocab = build_vocab_from_iterator(
            yield_trg(), min_freq=min_freq, specials=specials, special_first=True
        )
        src_vocab.set_default_index(src_vocab["<unk>"])
        trg_vocab.set_default_index(trg_vocab["<unk>"])
        self.source = _attach_legacy_vocab_attrs(src_vocab)
        self.target = _attach_legacy_vocab_attrs(trg_vocab)

    def make_iter(self, train, validate, test, batch_size, device):
        pad_src = self.source["<pad>"]

        train_ds = _TranslationDataset(train, self._src_tok, self._trg_tok, self.source, self.target)
        valid_ds = _TranslationDataset(validate, self._src_tok, self._trg_tok, self.source, self.target)
        test_ds = _TranslationDataset(test, self._src_tok, self._trg_tok, self.source, self.target)

        collate = _collate_batch(pad_src, device)
        train_iterator = TorchDataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
        )
        valid_iterator = TorchDataLoader(
            valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
        )
        test_iterator = TorchDataLoader(
            test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
        )
        print("dataset initializing done")
        return train_iterator, valid_iterator, test_iterator
