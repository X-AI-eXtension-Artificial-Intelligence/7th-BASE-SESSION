import spacy

class Tokenizer:

    def __init__(self):
        self.spacy_de = spacy.load('de_core_news_sm')       # 독일어 LM 로드
        self.spacy_en = spacy.load('en_core_web_sm')        # 영어 LM 로드

    def tokenize_de(self, text):

        return [tok.text for tok in self.spacy_de.tokenizer(text)]      # 독일어 텍스트를 토큰화하여 리스트로 반환

    def tokenize_en(self, text):

        return [tok.text for tok in self.spacy_en.tokenizer(text)]      # 영어 텍스트를 토큰화하여 리스트로 반환