from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import torch
import torchtext
torchtext.disable_torchtext_deprecation_warning()

def open_text_set(dir: str) -> list:
    #train, val 파일 오픈 용도
    with open(dir, 'r') as f:
        f = f.readlines()
        f = [v.strip() for v in f]
    return f

def make_vocab(
    src_lang: Iterable,
    trg_lang: Iterable,
) -> list:
    #vocab 생성 (src_lang : input 언어, trg_lang : output 언어)
    
    token_transform = {}
    vocab_transform = {}

    SRC_LANGUAGE, TRG_LANGUAGE = ["fr", "en"]
    # Load_tokenizer
    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='fr_core_news_sm')
    token_transform[TRG_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

    # token 생성 위한 iterator
    def yield_tokens(data_iter: Iterable, language: str) -> Iterable[str]:

        for data_sample in data_iter:
            yield token_transform[language](data_sample) # memory 초과 방지용

    # special tokens
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    
    for train_iter, ln in [(src_lang, SRC_LANGUAGE), (trg_lang, TRG_LANGUAGE)]:
        
        # 언어별 vocab 생성
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)
        
    for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX) # default index 설정 (unk 토큰)

    # vocab, tokenizer 저장
    return [vocab_transform, token_transform]


def sequential_transforms(*transforms):
    # *transform에 포함된 함수를 연속적으로 수행하게 하는 메서드

    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func

def tensor_transform(token_ids: List[int]):
    # 토크나이징 한 문장에 시작과 끝 의미하는 <bos>, <eos> 추가
    return torch.cat((torch.tensor([2]), torch.tensor(token_ids), torch.tensor([3])))