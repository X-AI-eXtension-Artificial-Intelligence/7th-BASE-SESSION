"""
util/data_loader.py
- torchtext의 Field, Multi30k, BucketIterator를 감싼 파일입니다.
- 원 코드는 torchtext.legacy API에 의존합니다.
"""

from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k


class DataLoader:
    # 클래스 변수로 선언되어 있지만, 실제로는 __init__/make_dataset 이후 인스턴스 변수처럼 사용됩니다.
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        # ext는 번역 방향을 결정합니다.
        # 예: ('.de', '.en')이면 독일어 → 영어입니다.
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self):
        """Multi30k dataset과 source/target Field를 생성합니다."""
        if self.ext == ('.de', '.en'):
            # source는 독일어, target은 영어입니다.
            self.source = Field(
                tokenize=self.tokenize_de,
                init_token=self.init_token,
                eos_token=self.eos_token,
                lower=True,
                batch_first=True
            )
            self.target = Field(
                tokenize=self.tokenize_en,
                init_token=self.init_token,
                eos_token=self.eos_token,
                lower=True,
                batch_first=True
            )

        elif self.ext == ('.en', '.de'):
            # source는 영어, target은 독일어입니다.
            self.source = Field(
                tokenize=self.tokenize_en,
                init_token=self.init_token,
                eos_token=self.eos_token,
                lower=True,
                batch_first=True
            )
            self.target = Field(
                tokenize=self.tokenize_de,
                init_token=self.init_token,
                eos_token=self.eos_token,
                lower=True,
                batch_first=True
            )

        # torchtext가 Multi30k train/valid/test split을 다운로드/로드합니다.
        train_data, valid_data, test_data = Multi30k.splits(
            exts=self.ext,
            fields=(self.source, self.target)
        )
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        """train 데이터 기준으로 source/target vocabulary를 만듭니다."""
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        """BucketIterator를 만들어 mini-batch 단위로 데이터를 공급합니다."""
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train, validate, test),
            batch_size=batch_size,
            device=device
        )
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator
