"""
@author : Hyunwoong
@when : 2019-10-29
torchtext.legacy + spacy 대신 공백 기반 토크나이저로 대체
"""


class Tokenizer:
    def tokenize_en(self, text):
        return text.lower().strip().split()

    def tokenize_de(self, text):
        return text.lower().strip().split()
