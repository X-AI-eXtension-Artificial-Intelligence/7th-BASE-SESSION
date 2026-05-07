from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

# 데이터 로더와 토크나이저 초기화
tokenizer = Tokenizer()
loader = DataLoader(ext=('.en', '.de'),                     # 데이터 파일의 확장자를 영어-독일어로 설정
                    tokenize_en=tokenizer.tokenize_en,      # 영어 텍스트를 토큰화하는 함수로 tokenizer의 tokenize_en 메서드를 사용
                    tokenize_de=tokenizer.tokenize_de,      # 독일어 텍스트를 토큰화하는 함수로 tokenizer의 tokenize_de 메서드를 사용
                    init_token='<sos>',                     # 문장의 시작을 나타내는 토큰으로 '<sos>'를 사용
                    eos_token='<eos>')                      # 문장의 끝을 나타내는 토큰으로 '<eos>'를 사용

train, valid, test = loader.make_dataset()           # 데이터셋을 생성하여 훈련, 검증, 테스트 데이터로 분할
loader.build_vocab(train_data=train, min_freq=2)     # 훈련 데이터에서 단어 집합을 구축 (최소 빈도 수를 2로 설정)   
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=batch_size,
                                                     device=device)     # 데이터셋을 배치 단위로 로드하여 훈련, 검증, 테스트 데이터 이터레이터를 생성 (배치 크기와 장치를 conf.py에서 설정한 값으로 사용)

src_pad_idx = loader.source.vocab.stoi['<pad>']     # 소스 언어의 패딩 토큰 인덱스를 가져옴
trg_pad_idx = loader.target.vocab.stoi['<pad>']     # 타겟 언어의 패딩 토큰 인덱스를 가져옴
trg_sos_idx = loader.target.vocab.stoi['<sos>']     # 타겟 언어의 문장 시작 토큰 인덱스를 가져옴

enc_voc_size = len(loader.source.vocab)     # 소스 언어의 단어 집합 크기를 계산하여 인코더의 어휘 크기로 설정
dec_voc_size = len(loader.target.vocab)     # 타겟 언어의 단어 집합 크기를 계산하여 디코더의 어휘 크기로 설정