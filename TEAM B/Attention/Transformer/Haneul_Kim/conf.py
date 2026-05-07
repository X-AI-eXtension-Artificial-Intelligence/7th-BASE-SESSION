"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 128
max_len = 256
d_model = 512 # 임베딩 차원
n_layers = 6 # Encoder/Decode layer 개수
n_heads = 8 # multi-head attention 개수
ffn_hidden = 2048 # FFN hidden dimension
drop_prob = 0.1 # 과적합 방지

# optimizer parameter setting
init_lr = 1e-5 # 초기 lr
factor = 0.9 # lr 감소 비율
adam_eps = 5e-9
patience = 10 # validation loss 개선 없을 때 몇 epoch 기다릴지
warmup = 100 # warmup epoch
epoch = 1000 # 전체 epoch 수
clip = 1.0 # gradient clipping 값
weight_decay = 5e-4
inf = float('inf')
