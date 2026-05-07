"""
util/tokenizer.py
- spaCy를 이용해 독일어/영어 문장을 토큰 리스트로 바꿉니다.
"""

import spacy


class Tokenizer:
    def __init__(self):
        # 독일어 tokenizer 모델입니다.
        # 실행 전: python -m spacy download de_core_news_sm
        self.spacy_de = spacy.load('de_core_news_sm')

        # 영어 tokenizer 모델입니다.
        # 실행 전: python -m spacy download en_core_web_sm
        self.spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(self, text):
        """독일어 문장을 토큰 문자열 리스트로 변환합니다."""
        # 예: "ein mann geht" -> ["ein", "mann", "geht"]
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """영어 문장을 토큰 문자열 리스트로 변환합니다."""
        # 예: "a man walks" -> ["a", "man", "walks"]
        return [tok.text for tok in self.spacy_en.tokenizer(text)]
