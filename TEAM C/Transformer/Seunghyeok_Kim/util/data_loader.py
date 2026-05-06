from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k

# 데이터 로더 클래스 정의
class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext                      # 데이터 파일의 확장자를 저장 (예: ('.de', '.en') 또는 ('.en', '.de'))
        self.tokenize_en = tokenize_en      # 영어 텍스트를 토큰화하는 함수
        self.tokenize_de = tokenize_de      # 독일어 텍스트를 토큰화하는 함수
        self.init_token = init_token        # 문장의 시작을 나타내는 토큰
        self.eos_token = eos_token          # 문장의 끝을 나타내는 토큰
        print('dataset initializing start')

    # 데이터셋을 생성하는 함수
    def make_dataset(self):
        if self.ext == ('.de', '.en'):      # 독일어를 소스 언어로, 영어를 타겟 언어로 설정
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)       # 독일어 텍스트를 토큰화하여 리스트로 반환하는 필드 설정
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)       # 영어 텍스트를 토큰화하여 리스트로 반환하는 필드 설정

        elif self.ext == ('.en', '.de'):        # 영어를 소스 언어로, 독일어를 타겟 언어로 설정
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)       # 영어 텍스트를 토큰화하여 리스트로 반환하는 필드 설정
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)       # 독일어 텍스트를 토큰화하여 리스트로 반환하는 필드 설정

        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))       # Multi30k 데이터셋을 로드하여 훈련, 검증, 테스트 데이터로 분할
        return train_data, valid_data, test_data

    # 단어 집합을 구축하는 함수
    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)      # 소스 언어의 단어 집합을 구축 (최소 빈도 수를 기준으로 단어를 포함)
        self.target.build_vocab(train_data, min_freq=min_freq)      # 타겟 언어의 단어 집합을 구축 (최소 빈도 수를 기준으로 단어를 포함)

    # 데이터셋을 배치 단위로 로드하는 함수
    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator