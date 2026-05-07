"""
data.py
- 토크나이저, 데이터셋, vocab, iterator를 생성하는 파일입니다.
- 이 파일이 실행되면 train.py에서 바로 사용할 수 있는 전역 변수들이 만들어집니다.
"""

from util.data_loader import DataLoader
from util.tokenizer import Tokenizer
from conf import *


# spaCy 기반 독일어/영어 토크나이저를 준비합니다.
# Tokenizer 안에서 de_core_news_sm, en_core_web_sm 모델을 로드합니다.
tokenizer = Tokenizer()

# DataLoader는 torchtext Field, Multi30k dataset, BucketIterator 생성을 감싼 클래스입니다.
loader = DataLoader(
    ext=ext,
    tokenize_en=tokenizer.tokenize_en,
    tokenize_de=tokenizer.tokenize_de,
    init_token=init_token,
    eos_token=eos_token
)

# Multi30k train/valid/test dataset을 만듭니다.
# ext가 ('.de', '.en')이면 source=독일어, target=영어입니다.
train, valid, test = loader.make_dataset()

# train 데이터 기준으로 source/target vocab을 만듭니다.
# validation/test에만 등장하는 단어는 vocab에 없으면 <unk>가 됩니다.
loader.build_vocab(train_data=train, min_freq=min_freq)

# BucketIterator는 길이가 비슷한 문장끼리 묶어 padding 낭비를 줄입니다.
train_iter, valid_iter, test_iter = loader.make_iter(
    train=train,
    validate=valid,
    test=test,
    batch_size=batch_size,
    device=device
)

# source/target vocab 크기입니다.
# Transformer의 embedding table 크기와 최종 classifier 출력 차원이 됩니다.
enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)

# padding token index입니다.
# mask 생성과 loss ignore_index에 사용됩니다.
src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']

# target 시작 토큰 index입니다.
# 현재 Transformer 클래스에서는 저장만 하고 직접 사용하지는 않습니다.
trg_sos_idx = loader.target.vocab.stoi['<sos>']
