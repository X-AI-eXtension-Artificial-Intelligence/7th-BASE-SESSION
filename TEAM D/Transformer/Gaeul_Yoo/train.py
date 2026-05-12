from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import Iterable, List
from data import fr_to_en
import util 
import torch.nn as nn
import pandas as pd
import json
import torch

## Vocab 만들기

fr_train = util.open_text_set("data/train/train.fr")
en_train = util.open_text_set("data/train/train.en")

try :
    vocab_transform, token_transform = util.make_vocab(fr_train, en_train)
except :
    import spacy.cli
    spacy.cli.download("fr_core_news_sm")
    spacy.cli.download("en_core_web_sm")
    vocab_transform, token_transform = util.make_vocab(fr_train, en_train)

# param
SRC_LANGUAGE, TRG_LANGUAGE = ["fr", "en"]


## Transformer

import math
import torch.nn.functional as F

class selfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(selfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        # 변수명 통일: _linear를 붙여서 명확하게 구분합니다.
        self.values_linear = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys_linear = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries_linear = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # N: Batch Size
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Multi-head로 쪼개기: (N, seq_len, heads, head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # 선형 변환 적용
        values = self.values_linear(values)
        keys = self.keys_linear(keys)
        queries = self.queries_linear(queries)

        # einsum을 이용한 행렬 곱 (Attention Score 계산)
        # n: batch, q: query_len, k: key_len, h: heads, d: head_dim
        # 결과: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            # 에러 해결의 핵심: mask의 차원을 (N, 1, 1, key_len)으로 맞춰줍니다.
            # 8(heads)과 31(seq_len)이 충돌하지 않도록 브로드캐스팅을 유도합니다.
            if mask.dim() == 2: # (N, key_len)인 경우
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3: # (N, 1, key_len)인 경우
                mask = mask.unsqueeze(1)
            
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Attention Weight 계산 및 적용
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        
        # 결과 결합: (N, query_len, heads, head_dim) -> (N, query_len, embed_size)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out
    
class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
        """
        config.json 참고

        embed_size(=512) : embedding 차원
        heads(=8) : Attention 개수
        dropout(=0.1): Node 학습 비율
        forward_expansion(=2) : FFNN의 차원을 얼마나 늘릴 것인지 결정,
                                forward_expension * embed_size(2*512 = 1024)
        """
        super().__init__()
        # Attention 정의
        self.attention = selfAttention(embed_size, heads)

        ### Norm & Feed Forward
        self.norm1 = nn.LayerNorm(embed_size)  # 512
        self.norm2 = nn.LayerNorm(embed_size)  # 512

        self.feed_forawrd = nn.Sequential(
            # 512 => 1024
            nn.Linear(embed_size, forward_expansion * embed_size),
            # ReLU 연산
            nn.ReLU(),
            # 1024 => 512
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):

        # self Attention
        attention = self.attention(value, key, query, mask)
        # Add & Normalization
        x = self.dropout(self.norm1(attention + query))
        # Feed_Forward
        forward = self.feed_forawrd(x)
        # Add & Normalization
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
        device,
    ) -> None:
        """
        config.json 참고

        src_vocab_size(=11509) : input vocab 개수
        embed_size(=512) : embedding 차원
        num_layers(=3) : Encoder Block 개수
        heads(=8) : Attention 개수
        device : cpu;
        forward_expansion(=2) : FFNN의 차원을 얼마나 늘릴 것인지 결정,
                                forward_expension * embed_size(2*512 = 1024)
        dropout(=0.1): Node 학습 비율
        max_length : batch 문장 내 최대 token 개수(src_token_len)
        """
        super().__init__()
        self.embed_size = embed_size
        self.device = device

        # input + positional_embeding
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)  # (11509, 512) 2

        # positional embedding
        pos_embed = torch.zeros(max_length, embed_size)  # (src_token_len, 512) 2
        pos_embed.requires_grad = False
        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed = pos_embed.unsqueeze(0).to(device)  # (1, src_token_len, 512) 3

        # Encoder Layer 구현
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        _, seq_len = x.size()  # (n, src_token_len) 2
        # n : batch_size(=128)
        # src_token_len : batch 내 문장 중 최대 토큰 개수

        pos_embed = self.pos_embed[:, :seq_len, :]
        # (1, src_token_len, embed_size) 3

        out = self.dropout(self.word_embedding(x) + pos_embed)
        # (n, src_token_len, embed_size) 3

        for layer in self.layers:
            # Q,K,V,mask
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
        """
        config.json 참고

        embed_size(=512) : embedding 차원
        heads(=8) : Attention 개수
        dropout(=0.1): Node 학습 비율
        forward_expansion(=2) : FFNN의 차원을 얼마나 늘릴 것인지 결정,
                                forward_expension * embed_size(2*512 = 1024)
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = selfAttention(embed_size, heads=heads)
        self.encoder_block = EncoderBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_trg_mask, target_mask):
        """
        x : target input with_embedding (n, trg_token_len, embed_size) 3
        value, key : encoder_attention (n, src_token_len, embed_size) 3
        """

        # masked_attention
        attention = self.attention(x, x, x, target_mask)
        # (n, trg_token_len, embed_size) 3

        # add & Norm
        query = self.dropout(self.norm(attention + x))

        # encoder_decoder attention + feed_forward
        out = self.encoder_block(value, key, query, src_trg_mask)
        # (n, trg_token_len, embed_size) 3

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
        device,
    ) -> None:
        """
        config.json 참고

        trg_vocab_size(=10873) : input vocab 개수
        embed_size(=512) : embedding 차원
        num_layers(=3) : Encoder Block 개수
        heads(=8) : Attention 개수
        forward_expansion(=2) : FFNN의 차원을 얼마나 늘릴 것인지 결정,
                                forward_expension * embed_size(2*512 = 1024)
        dropout(=0.1): Node 학습 비율
        max_length : batch 문장 내 최대 token 개수
        device : cpu
        """
        super().__init__()
        self.device = device

        # 시작부분 구현(input + positional_embeding)
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)  # (10837,512) 2

        # positional embedding
        pos_embed = torch.zeros(max_length, embed_size)  # (trg_token_len, embed_size) 2
        pos_embed.requires_grad = False
        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed = pos_embed.unsqueeze(0).to(device)
        # (1, trg_token_len, embed_size) 3

        # Decoder Layer 구현
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_trg_mask, trg_mask):
        # n : batch_size(=128)
        # trg_token_len : batch 내 문장 중 최대 토큰 개수

        _, seq_len = x.size()
        # (n, trg_token_len)

        pos_embed = self.pos_embed[:, :seq_len, :]
        # (1, trg_token_len, embed_size) 3

        out = self.dropout(self.word_embedding(x) + pos_embed).to(self.device)
        # (n, trg_token_len, embed_size) 3

        for layer in self.layers:
            # Decoder Input, Encoder(K), Encoder(V) , src_trg_mask, trg_mask
            out = layer(out, enc_out, enc_out, src_trg_mask, trg_mask)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size,
        num_layers,
        forward_expansion,
        heads,
        dropout,
        device,
        max_length,
    ) -> None:
        """
        src_vocab_size(=11509) : source vocab 개수
        trg_vocab_size(=10873) : target vocab 개수
        src_pad_idx(=1) : source vocab의 <pad> idx
        trg_pad_idx(=1) : source vocab의 <pad> idx
        embed_size(=512) : embedding 차원
        num_layers(=3) : Encoder Block 개수
        forward_expansion(=2) : FFNN의 차원을 얼마나 늘릴 것인지 결정,
                                forward_expension * embed_size(2*512 = 1024)
        heads(=8) : Attention 개수
        dropout(=0.1): Node 학습 비율
        device : cpu
        max_length(=140) : batch 문장 내 최대 token 개수

        """
        super().__init__()
        self.Encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
            device,
        )
        self.Decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
            device,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        # Probability Generlator
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)  # (512,10873) 2

    def encode(self, src):
        """
        Test 용도로 활용 encoder 기능
        """
        src_mask = self.make_pad_mask(src, src)
        return self.Encoder(src, src_mask)

    def decode(self, src, trg, enc_src):
        """
        Test 용도로 활용 decoder 기능
        """
        # decode
        src_trg_mask = self.make_pad_mask(trg, src)
        trg_mask = self.make_trg_mask(trg)
        out = self.Decoder(trg, enc_src, src_trg_mask, trg_mask)
        # Linear Layer
        out = self.fc_out(out)  # (n, decoder_query_len, trg_vocab_size) 3

        # Softmax
        out = F.log_softmax(out, dim=-1)
        return out

    def make_pad_mask(self, query, key):
        """
        Multi-head attention pad 함수
        """
        len_query, len_key = query.size(1), key.size(1)

        key = key.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (batch_size x 1 x 1 x src_token_len) 4

        key = key.repeat(1, 1, len_query, 1)
        # (batch_size x 1 x len_query x src_token_len) 4

        query = query.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        # (batch_size x 1 x src_token_len x 1) 4

        query = query.repeat(1, 1, 1, len_key)
        # (batch_size x 1 x src_token_len x src_token_len) 4

        mask = key & query
        return mask

    def make_trg_mask(self, trg):
        """
        Masked Multi-head attention pad 함수
        """
        # trg = triangle
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src)
        # (n,1,src_token_len,src_token_len) 4

        trg_mask = self.make_trg_mask(trg)
        # (n,1,trg_token_len,trg_token_len) 4

        src_trg_mask = self.make_pad_mask(trg, src)
        # (n,1,trg_token_len,src_token_len) 4

        enc_src = self.Encoder(src, src_mask)
        # (n, src_token_len, embed_size) 3

        out = self.Decoder(trg, enc_src, src_trg_mask, trg_mask)
        # (n, trg_token_len, embed_size) 3

        # Linear Layer
        out = self.fc_out(out)  # embed_size => trg_vocab_size
        # (n, trg_token_len, trg_vocab_size) 3

        # Softmax
        out = F.log_softmax(out, dim=-1)
        return out

## 모델 불러오기
with open('config/transformer.json', 'r') as file:
    param = json.load(file)
    print('Model_Parameter')
    print(param)    
model = Transformer(**param)
device = param['device']

print('-'*50)
print(f'현재 devicde 설정값은 : "{device}" 입니다. 변경을 희망하실 경우 config/transformer.json을 수정해주세요.')

# xavier 
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# loss_fn
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)

# optimzer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


## Training, Validation Loop

def collate_fn(batch_iter: Iterable):
    """
    Data_loader에서 불러온 데이터를 가공하는 함수
    토크나이징 => encoding => 시작 끝을 의미하는 spectial token(<bos>,<eos>) 추가 순으로 진행
    """
    text_transform = {}
    for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
        text_transform[ln] = util.sequential_transforms(
            token_transform[ln],  # 토크나이징
            vocab_transform[ln],  # encoding
            util.tensor_transform, # BOS/EOS를 추가하고 텐서를 생성
        )  
        # sequential_transform, tensor_transform은 utils.py 참고
    
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch_iter:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample))
        tgt_batch.append(text_transform[TRG_LANGUAGE](tgt_sample))

    # Pad 붙이기
    PAD_IDX = 1
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch.T, tgt_batch.T




# token을 단어로 바꾸기 위한 dict 생성, vocab의 key와 value 위치 변경
# 아래 helper 함수에서 활용됨.
decoder_en = {v:k for k,v in vocab_transform['en'].get_stoi().items()}
decoder_fr = {v:k for k,v in vocab_transform['fr'].get_stoi().items()}


def helper_what_sen(src,trg,logits,i,c=100,sen_num=0) : 
    '''
    문장이 제대로 학습되고 있는지를 확인하는 함수

    src = encoding 된 source_sentence 
    trg = encoding 된 target_sentence
    logits = 모델 예측값
    i = 현재 batch 순서
    c = 결과를 보여주는 단계, ex) c = 100이면 100,200,300... 번째 batch에서 결과를 보여줌
    sen_num = batch 내 문장 중 몇 번째 문장을 추적할 것인지 설정
    '''
    if i % c == 0 and i != 0 :
        src_sen = ' '.join([decoder_fr[i] for i in src.tolist()[sen_num] if decoder_fr[i][0] != '<' ])
        trg_sen = ' '.join([decoder_en[i] for i in trg.tolist()[sen_num] if decoder_en[i][0] != '<' ])
        prediction = logits.max(dim=-1, keepdim=False)[1][sen_num]
        prd_sen = ' '.join([decoder_en[i] for i in prediction.tolist() if decoder_en[i] != '<' ])
        '''
        /*/* 모델의 예측 문장(prd_sen)을 구하는 방법 /*/* 

        n = batch size, trg_token_len = batch 내 문장의 최대 토큰 개수

        모델 output(=logits)은 (n, trg_token_len, trg_vocab_len)의 3차원 텐서임.

        1. 해당 텐서를 trg_vocab_len 차원의 기준으로 max를 하면 (n,trg_token_len)을 반환
        2. tensor.max()의 수행 결과는 [최댓값,idx]를 반환함.
        3. [1]을 넣어 idx를 선택, 그 결과는 (n, trg_token_len) 차원의 idx 반환
        4. 원하는 문장 순서(sen_num)을 선택한 뒤 정수를 다시 단어로 decoding 수행
        

        '''

        print('')
        print(f'{i}번째 batch에 있는 {sen_num}번째 문장 예측 결과 확인')
        print('src : ',src_sen)
        print('prd : ',prd_sen)
        print('trg : ',trg_sen)
        print('')

    return None


def train_epoch(model,optimizer) : 
    '''
    훈련 과정 설정
    '''
    # 모델 훈련 모드 변경
    model.train()
    losses = 0

    # training 데이터 불러오기(29,000개 문장)
    dataset= fr_to_en(set_type='train')

    # Data_loader
    batch_size = 128
    train_dataloader = DataLoader(dataset,batch_size,collate_fn=collate_fn)

    # 128개 문장을 1개의 batch로 총 227번 반복(128*227 => 29,000)

    for i,(src,tgt) in enumerate(train_dataloader) :
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:,:-1] # 마지막 token은 <eos> 또는 <pad>이므로 불필요한 데이터 전처리

        logits = model(src,tgt_input) # (n, trg_token_len, trg_vocab_len) 3

        optimizer.zero_grad()

        helper_what_sen(src,tgt_input,logits,i,200,0) # 학습상태 확인

        tgt_output = tgt[:,1:] # 첫번째 token은 <bos>이므로 불필요한 데이터 전처리
        loss = loss_fn(logits.reshape(-1,logits.shape[-1]),tgt_output.reshape(-1))
        '''
        */*/*/*/ loss_fn(model_output,target_out_put) 설명 */*/*/*/*/*/

        model_output은 (n, trg_token_len, trg_vocab_len)의 3차원 텐서임 
        target_output은 (n,trg_token_len)의 2차원 텐서임
        이를 각각 2차원, 1차원으로 줄이면 
        model_output = (n*trg_token_len, trg_vocab_len)
        target_output = (n*trg_token_len) 

        이때 target_output은 개별 단어의 idx이므로 
        model_output의 model_outpu의 idx로 활용하면 해당 단어의 확률을 찾을 수 있음.
        
        이러한 방법으로 정답인 단어와, 정답이 아닌 단어를 구분하여 Loss 계산 수행
        '''

        loss.backward()

        optimizer.step()

        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    #모델 평가모드 
    model.eval()
    losses = 0
    
    # Load_Dataset
    dataset= fr_to_en(set_type='valid')

    # validation 데이터 불러오기
    batch_size = 128
    val_dataloader = DataLoader(dataset,batch_size,collate_fn=collate_fn)

    for i,(src,tgt) in enumerate(val_dataloader) :
        
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:,:-1]

        logits = model(src,tgt_input)

        helper_what_sen(src,tgt_input,logits,i,5,0) # 학습상태 확인

        tgt_output = tgt[:,1:]
        loss = loss_fn(logits.reshape(-1,logits.shape[-1]),tgt_output.reshape(-1))

        losses += loss.item()

    return losses / len(val_dataloader)

    ## 학습 및 평가 진행

from timeit import default_timer as timer
NUM_EPOCHS = 10

for epoch in range(1, NUM_EPOCHS+1):
    print('-'*30)
    print(f'{epoch}번째 epoch 실행')
    start_time = timer()
    train_loss = train_epoch(model.to(device), optimizer)
    end_time = timer()
    val_loss = evaluate(model)
    

    print('----*'*20)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    print('----*'*20)
    print('')
## 학습 및 평가 진행
from timeit import default_timer as timer
import os

# 모델 저장 폴더 생성
if not os.path.exists('model'):
    os.makedirs('model')

NUM_EPOCHS = 10

for epoch in range(1, NUM_EPOCHS+1):
    print('-'*30)
    print(f'{epoch}번째 epoch 실행')
    
    start_time = timer()
    train_loss = train_epoch(model.to(device), optimizer)
    end_time = timer()
    
    val_loss = evaluate(model)
    
    # [수정 포인트] 매 에폭이 끝날 때마다 모델 저장
    # 혹은 학습이 완전히 끝난 후 한 번만 저장해도 됩니다.
    torch.save(model.state_dict(), 'model/model.pth')
    print(f'>>> Epoch {epoch} 완료: model/model.pth 저장됨')

    print('----*'*20)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    print('----*'*20)
    print('')

print("최종 학습 완료 및 모델 저장 성공!")


