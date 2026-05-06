"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

# Tokenizer 객체 생성
# 영어 문장과 독일어 문장을 각각 토큰화하는 함수가 들어 있음
tokenizer = Tokenizer()
loader = DataLoader(
                    # 사용할 데이터 파일 확장자 지정
                    # source 문장은 .en, target 문장은 .de 파일에서 읽어옴
                    ext=('.en', '.de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')
# train, valid, test dataset 생성
train, valid, test = loader.make_dataset()

# 학습 데이터 train을 기준으로 vocabulary 생성
# min_freq=2는 최소 2번 이상 등장한 단어만 vocabulary에 포함한다는 뜻
loader.build_vocab(train_data=train, min_freq=2)
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=batch_size,
                                                     device=device)
# source 문장에서 padding token의 index를 가져옴
# padding token은 길이가 다른 문장들을 같은 길이로 맞추기 위해 사용됨
src_pad_idx = loader.source.vocab.stoi['<pad>']
# target 문장에서 padding token의 index를 가져옴
# loss 계산 시 padding 부분은 무시하기 위해 필요함
trg_pad_idx = loader.target.vocab.stoi['<pad>']
# target 문장에서 <sos> token의 index를 가져옴
# decoder가 문장을 생성할 때 시작 토큰으로 사용함
trg_sos_idx = loader.target.vocab.stoi['<sos>']

# encoder vocabulary 크기
# 즉 source 언어, 여기서는 영어 단어장의 크기
enc_voc_size = len(loader.source.vocab)
# decoder vocabulary 크기
# 즉 target 언어, 여기서는 독일어 단어장의 크기
dec_voc_size = len(loader.target.vocab)
