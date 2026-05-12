from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import Iterable, List
from model import Transformer
from data import fr_to_en
import util
import torch.nn as nn
import pandas as pd
import json
import torch

# 훈련 데이터 불러오기
fr_data = util.open_text_set("data/train/train.fr")
eng_data = util.open_text_set("data/train/train.en")

# Vocab 만들기
try:
    vocab_transform, token_transform = util.make_vocab(fr_data, eng_data)
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    spacy.cli.download("fr_core_news_sm")
    vocab_transform, token_transform = util.make_vocab(fr_data, eng_data)

# 특수 토큰 설정
BOS_IDX = vocab_transform['en']['<bos>']
EOS_IDX = vocab_transform['en']['<eos>']
PAD_IDX = vocab_transform['en']['<pad>']

with open('config/transformer.json', 'r') as file:
    param = json.load(file)
    print('Model_Parameters')
    print('-'*50)
    print(param)

# 모델 불러오기
model = Transformer(**param)
model.load_state_dict(torch.load('model/model.pth', map_location=torch.device(param.get('device', 'cpu'))))
model.to(param.get('device', 'cpu'))
model.eval()

device = param.get('device', 'cpu')

print('-'*50)
print(f'현재 device 설정값은 : "{device}" 입니다.')
print('-'*50)

decoder_en = {v: k for k, v in vocab_transform['en'].get_stoi().items()}

def tokenizing_src(input_data: str):
    token_data = token_transform['fr'](input_data)
    vocab_src = vocab_transform['fr'](token_data)
    # [BOS] + tokens + [EOS]
    tokenized_src = [vocab_transform['fr']['<bos>']] + vocab_src + [vocab_transform['fr']['<eos>']]
    return tokenized_src

def select_random_item():
    num = torch.randint(0, len(fr_data), (1,)).item()
    # 수정: 두 번째 반환값을 eng_data로 변경
    return fr_data[num], eng_data[num]

def test(model):
    model.eval()

    # 임의의 데이터 선별
    fr_item, en_item = select_random_item()
    print('입력(FR) :', fr_item)

    tokenized_input = tokenizing_src(fr_item)
    # max_length는 config 혹은 입력 길이에 비례하게 설정
    max_length = param.get('max_length', 140)

    src = torch.LongTensor(tokenized_input).unsqueeze(0).to(device)
    
    # Encoder 연산은 한 번만 수행
    enc_src = model.encode(src)

    # 시작 토큰 세팅
    trg_indexes = [BOS_IDX]

    # 문장 예측 루프
    for i in range(max_length):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        # 현재까지의 trg를 바탕으로 다음 단어 예측
        with torch.no_grad():
            logits = model.decode(src, trg_tensor, enc_src)
        
        # 마지막 타임스텝의 결과물에서 가장 확률 높은 단어 선택
        prd = logits.squeeze(0).max(dim=-1)[1]
        next_word = prd[-1].item() # 마지막 예측 토큰 추출

        trg_indexes.append(next_word)

        if next_word == EOS_IDX:
            break
    
    # 번역 결과 디코딩 (BOS, EOS 제외)
    translation = [decoder_en[idx] for idx in trg_indexes if idx not in [BOS_IDX, EOS_IDX]]
    
    print('모델예측(EN) :', ' '.join(translation))
    print('실제정답(EN) :', en_item)
    print('-'*50)
    print('주의! 29,000개의 제한된 데이터로 학습을 수행했으므로 완벽한 예측이 불가능함.')

# 실행
test(model)