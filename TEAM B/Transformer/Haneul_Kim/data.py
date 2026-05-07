"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

# tokenizer 객체 생성
# 영어 / 독일어 문장을 토큰 단위로 분리
tokenizer = Tokenizer()
loader = DataLoader(ext=('.en', '.de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')

# train / validation / test 데이터셋 생성
train, valid, test = loader.make_dataset()

# 단어 사전(vocab) 생성
# min_freq=2 -> 2번 이상 등장한 단어만 vocab에 포함
loader.build_vocab(train_data=train, min_freq=2)

# iterator 생성
# batch 단위로 데이터 로딩
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=batch_size,
                                                     device=device)

# padding token index
src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']
# 시작 토큰 index
trg_sos_idx = loader.target.vocab.stoi['<sos>']

# source / target vocab 크기
enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)
