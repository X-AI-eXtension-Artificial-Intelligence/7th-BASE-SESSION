import torch
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def build_data(batch_size):
    # 토크나이저 설정
    token_transform = {}
    token_transform['src'] = get_tokenizer('spacy', language='fr_core_news_sm')
    token_transform['trg'] = get_tokenizer('spacy', language='en_core_web_sm')

    def yield_tokens(data_iter, language):
        for data_sample in data_iter:
            yield token_transform[language](data_sample[0 if language == 'src' else 1])

    # 보카브러리 생성
    vocab_transform = {}
    for ln in ['src', 'trg']:
        train_iter = Multi30k(split='train', language_pair=('fr', 'en'))
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                      min_freq=1, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        vocab_transform[ln].set_default_index(vocab_transform[ln]['<unk>'])

    # 데이터 전처리 함수
    def collate_fn(batch):
        src_batch, trg_batch = [], []
        for src_sample, trg_sample in batch:
            src_batch.append(torch.tensor(vocab_transform['src'](token_transform['src'](src_sample))))
            trg_batch.append(torch.tensor(vocab_transform['trg'](token_transform['trg'](trg_sample))))
        
        src_batch = pad_sequence(src_batch, padding_value=vocab_transform['src']['<pad>'], batch_first=True)
        trg_batch = pad_sequence(trg_batch, padding_value=vocab_transform['trg']['<pad>'], batch_first=True)
        return src_batch, trg_batch

    train_iter = Multi30k(split='train', language_pair=('fr', 'en'))
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)
    
    return train_dataloader, vocab_transform